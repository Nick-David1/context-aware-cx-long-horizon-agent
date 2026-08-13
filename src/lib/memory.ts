import { randomUUID } from "node:crypto";
import { collections, VECTOR_INDEX } from "./mongo";
import { embedOne, rerank } from "./embeddings";
import type { Memory, MemoryKind } from "./types";

/**
 * The context engine.
 *
 * Every inference pulls from three different slices of memory and fuses them:
 *   policy   — authored business logic, global
 *   episodic — what actually happened with this specific customer
 *   lesson   — generalizations the agent derived from past outcomes
 *
 * They are separate searches rather than one, because a single top-k over the
 * whole collection lets a chatty customer's episodic notes crowd out the policy
 * that governs the action. Slice quotas keep each type represented.
 */

export interface RecalledMemory {
  memoryId: string;
  kind: MemoryKind;
  text: string;
  /** Cosine similarity from $vectorSearch. */
  score: number;
  /** Cross-encoder relevance from the reranker, when it ran. */
  rerankScore?: number;
  /** Lesson win rate, when known. Drives ordering and is shown in the UI. */
  winRate?: number;
}

/**
 * How many candidates to pull per slice before reranking. The reranker is only
 * as good as the shortlist it sees, and $vectorSearch is cheap — so fetch wide,
 * then let the cross-encoder cut it down to the quota.
 */
const OVERFETCH = 4;

interface SliceSpec {
  kind: MemoryKind;
  customerId: string | null;
  limit: number;
}

async function searchSlice(
  queryVector: number[],
  spec: SliceSpec,
  tags: string[],
): Promise<RecalledMemory[]> {
  const { memories } = await collections();

  const filter: Record<string, unknown> = {
    active: true,
    kind: spec.kind,
    customerId: spec.customerId,
  };
  if (tags.length > 0) filter.tags = { $in: tags };

  const docs = await memories
    .aggregate<Memory & { score: number }>([
      {
        $vectorSearch: {
          index: VECTOR_INDEX,
          path: "embedding",
          queryVector,
          // Over-fetch candidates so the pre-filter has room to work.
          numCandidates: Math.max(spec.limit * 20, 100),
          limit: spec.limit,
          filter,
        },
      },
      { $addFields: { score: { $meta: "vectorSearchScore" } } },
      { $project: { embedding: 0 } },
    ])
    .toArray();

  return docs.map((d) => ({
    memoryId: d.memoryId,
    kind: d.kind,
    text: d.text,
    score: d.score,
    winRate: winRate(d),
  }));
}

function winRate(m: Pick<Memory, "stats">): number | undefined {
  const s = m.stats;
  if (!s || s.timesApplied === 0) return undefined;
  return s.wins / s.timesApplied;
}

export interface RecallOptions {
  query: string;
  customerId: string;
  tags?: string[];
  policyLimit?: number;
  episodicLimit?: number;
  lessonLimit?: number;
}

export async function recall(opts: RecallOptions): Promise<RecalledMemory[]> {
  const {
    query,
    customerId,
    tags = [],
    policyLimit = 5,
    episodicLimit = 5,
    lessonLimit = 4,
  } = opts;

  const queryVector = await embedOne(query, "query");

  // Stage 1 — recall. Over-fetch each slice with $vectorSearch.
  const [policies, episodic, lessons] = await Promise.all([
    searchSlice(
      queryVector,
      { kind: "policy", customerId: null, limit: policyLimit * OVERFETCH },
      tags,
    ),
    searchSlice(
      queryVector,
      { kind: "episodic", customerId, limit: episodicLimit * OVERFETCH },
      [],
    ),
    searchSlice(
      queryVector,
      { kind: "lesson", customerId: null, limit: lessonLimit * OVERFETCH },
      tags,
    ),
  ]);

  // Stage 2 — precision. One rerank call over the whole shortlist, then apply
  // per-slice quotas to the reranked order. Reranking across slices rather than
  // within each one lets the cross-encoder compare a policy against a lesson,
  // while the quotas still guarantee every slice is represented.
  const candidates = [...policies, ...episodic, ...lessons];
  const hits = await rerank(query, candidates.map((c) => c.text));

  let ordered = candidates;
  if (hits) {
    for (const hit of hits) candidates[hit.index].rerankScore = hit.relevanceScore;
    ordered = hits.map((h) => candidates[h.index]);
  }

  // Stage 3 — quotas and lesson credit. A lesson that has lost more than it won
  // is actively misleading, so drop it and let reflection replace it.
  const quota: Record<MemoryKind, number> = {
    policy: policyLimit,
    episodic: episodicLimit,
    lesson: lessonLimit,
  };
  const taken: Record<MemoryKind, number> = { policy: 0, episodic: 0, lesson: 0 };

  const selected: RecalledMemory[] = [];
  for (const m of ordered) {
    if (taken[m.kind] >= quota[m.kind]) continue;
    if (m.kind === "lesson" && m.winRate !== undefined && m.winRate < 0.4) continue;
    taken[m.kind] += 1;
    selected.push(m);
  }

  // Group by kind for a stable, readable context block.
  const order: MemoryKind[] = ["policy", "episodic", "lesson"];
  return selected.sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind));
}

/** Renders recalled memories into the block that goes into the model's system prompt. */
export function renderContext(memories: RecalledMemory[]): string {
  if (memories.length === 0) return "(no relevant context found)";
  const group = (kind: MemoryKind, heading: string) => {
    const items = memories.filter((m) => m.kind === kind);
    if (items.length === 0) return "";
    const lines = items.map((m) => {
      const rate =
        m.winRate === undefined ? "" : ` [win rate ${(m.winRate * 100).toFixed(0)}%]`;
      return `- (${m.memoryId})${rate} ${m.text}`;
    });
    return `${heading}\n${lines.join("\n")}\n`;
  };
  return [
    group("policy", "## Company policy"),
    group("episodic", "## What we know about this customer"),
    group("lesson", "## Lessons learned from past cases"),
  ]
    .filter(Boolean)
    .join("\n");
}

export interface RememberInput {
  kind: MemoryKind;
  customerId: string | null;
  text: string;
  tags?: string[];
  sourceCaseIds?: string[];
  /** memoryId of a memory this one replaces. */
  supersedes?: string;
}

export async function remember(input: RememberInput): Promise<Memory> {
  const { memories } = await collections();
  const embedding = await embedOne(input.text, "document");
  const now = new Date();

  const doc: Memory = {
    memoryId: randomUUID(),
    kind: input.kind,
    customerId: input.customerId,
    text: input.text,
    embedding,
    tags: input.tags ?? [],
    active: true,
    stats: input.kind === "lesson" ? { timesApplied: 0, wins: 0, losses: 0 } : undefined,
    sourceCaseIds: input.sourceCaseIds,
    createdAt: now,
    updatedAt: now,
  };

  await memories.insertOne(doc);

  if (input.supersedes) {
    await memories.updateOne(
      { memoryId: input.supersedes },
      { $set: { active: false, supersededBy: doc.memoryId, updatedAt: now } },
    );
  }

  return doc;
}

/**
 * Credit assignment: when a case closes, every lesson that was in context gets
 * its win/loss counters bumped. This is what makes retrieval improve over time
 * rather than just accumulate.
 */
export async function scoreLessons(
  memoryIds: string[],
  result: "won" | "lost",
): Promise<void> {
  if (memoryIds.length === 0) return;
  const { memories } = await collections();
  await memories.updateMany(
    { memoryId: { $in: memoryIds }, kind: "lesson" },
    {
      $inc: {
        "stats.timesApplied": 1,
        "stats.wins": result === "won" ? 1 : 0,
        "stats.losses": result === "lost" ? 1 : 0,
      },
      $set: { updatedAt: new Date() },
    },
  );
}
