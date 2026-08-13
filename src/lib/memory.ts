import { randomUUID } from "node:crypto";
import { collections, VECTOR_INDEX } from "./mongo";
import { embedOne } from "./embeddings";
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
  score: number;
  /** Lesson win rate, when known. Drives re-ranking and is shown in the UI. */
  winRate?: number;
}

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

  const [policies, episodic, lessons] = await Promise.all([
    searchSlice(queryVector, { kind: "policy", customerId: null, limit: policyLimit }, tags),
    searchSlice(
      queryVector,
      { kind: "episodic", customerId, limit: episodicLimit },
      [],
    ),
    searchSlice(queryVector, { kind: "lesson", customerId: null, limit: lessonLimit }, tags),
  ]);

  // A lesson that has lost more than it won is actively misleading — drop it and
  // let reflection replace it. Otherwise prefer proven lessons.
  const rankedLessons = lessons
    .filter((l) => l.winRate === undefined || l.winRate >= 0.4)
    .sort((a, b) => (b.winRate ?? 0.5) - (a.winRate ?? 0.5));

  return [...policies, ...episodic, ...rankedLessons];
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
