import { randomUUID } from "node:crypto";
import { collections } from "./mongo";
import { config } from "./config";
import { embedOne, maybeEmbed, rerank } from "./embeddings";
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
 *
 * Each slice is two-stage: $vectorSearch over-fetches, then a cross-encoder
 * reranks down to the quota. Vector search embeds query and document
 * separately, which makes it fast over the whole collection and blunt at the
 * top of the list; the cross-encoder reads the pair together and fixes exactly
 * that. Reranking runs inside the aggregation via $rerank where the cluster
 * supports it, and client-side otherwise.
 */

export interface RecalledMemory {
  memoryId: string;
  kind: MemoryKind;
  text: string;
  /** Vector similarity, or the reranked score once reranking has run. */
  score: number;
  /** Cross-encoder relevance, when reranking ran. */
  rerankScore?: number;
  /** Lesson win rate, when known. Drives filtering and is shown in the UI. */
  winRate?: number;
}

/**
 * How many candidates to pull per slice before reranking. The cross-encoder can
 * only promote what retrieval handed it, so fetching exactly the quota and then
 * reranking reorders without improving anything.
 */
const OVERFETCH = 4;

/** Set once if the cluster rejects a stage, so we stop trying. */
let nativeRerankDowngraded = false;
let hybridDowngraded = false;

export function rerankPath(): "native" | "client" | "off" {
  if (config.rerankMode === "off") return "off";
  if (config.rerankMode === "native" && !nativeRerankDowngraded) return "native";
  return "client";
}

export function retrievalPath(): "hybrid" | "vector" {
  return config.retrievalMode === "hybrid" && !hybridDowngraded ? "hybrid" : "vector";
}

function unsupportedStage(err: unknown, stage: string): boolean {
  const message = (err as Error)?.message?.toLowerCase() ?? "";
  return (
    message.includes(stage) ||
    message.includes("unrecognized pipeline stage") ||
    message.includes("unknown stage")
  );
}

interface SliceSpec {
  kind: MemoryKind;
  customerId: string | null;
  limit: number;
}

async function searchSlice(
  query: string,
  queryVector: number[] | null,
  spec: SliceSpec,
  tags: string[],
): Promise<RecalledMemory[]> {
  const { memories } = await collections();

  const scope = spec.customerId ?? "global";
  const overfetch = spec.limit * OVERFETCH;
  const useNative = rerankPath() === "native";
  const useHybrid = retrievalPath() === "hybrid";

  // $vectorSearch takes an MQL-subset filter.
  const vectorFilter: Record<string, unknown> = { active: true, kind: spec.kind, scope };
  if (tags.length > 0) vectorFilter.tags = { $in: tags };

  // $search takes Lucene compound clauses over token/boolean fields.
  const searchFilters: Record<string, unknown>[] = [
    { equals: { path: "active", value: true } },
    { equals: { path: "kind", value: spec.kind } },
    { equals: { path: "scope", value: scope } },
  ];
  if (tags.length > 0) searchFilters.push({ in: { path: "tags", value: tags } });

  const vectorPipeline = [
    {
      $vectorSearch: {
        index: config.vectorIndex,
        // Auto mode targets the text field the autoEmbed index reads and hands
        // MongoDB the raw query string; client mode targets the stored vector
        // and hands it a precomputed one. Everything downstream is identical.
        ...(config.embeddingMode === "auto"
          ? { path: config.embedTextField, query }
          : { path: "embedding", queryVector }),
        // Over-fetch candidates: our filters are always selective (kind +
        // customer scope), and selective pre-filters are exactly where HNSW
        // recall degrades if the candidate pool is tight.
        numCandidates: Math.max(overfetch * 20, 200),
        limit: overfetch,
        filter: vectorFilter,
      },
    },
  ];

  // The lexical half. Embeddings blur exact tokens — loan ids, "90 days",
  // dollar figures — and BM25 up-weights precisely those rare terms.
  const textPipeline = [
    {
      $search: {
        index: config.textIndex,
        compound: {
          filter: searchFilters,
          must: [{ text: { query, path: config.embedTextField } }],
        },
      },
    },
    { $limit: overfetch },
  ];

  const head = useHybrid
    ? [
        {
          // Reciprocal rank fusion over both retrievers, as one stage. Merges
          // the two rankings by position rather than by raw score, so a BM25
          // score and a cosine score never have to be made commensurate.
          $rankFusion: {
            input: { pipelines: { vector: vectorPipeline, text: textPipeline } },
            combination: { weights: config.fusionWeights },
          },
        },
        { $addFields: { score: { $meta: "score" } } },
      ]
    : [
        ...vectorPipeline,
        // Materialize the score before $rerank replaces $meta.
        { $addFields: { score: { $meta: "vectorSearchScore" } } },
      ];

  const pipeline: Record<string, unknown>[] = [...head];

  if (useNative) {
    pipeline.push(
      {
        $rerank: {
          query: { text: query },
          path: [config.embedTextField],
          numDocsToRerank: overfetch,
          model: config.rerankModel,
        },
      },
      { $addFields: { rerankScore: { $meta: "score" } } },
      { $addFields: { score: "$rerankScore" } },
      { $limit: spec.limit },
    );
  }

  pipeline.push({ $project: { embedding: 0 } });

  let docs: (Memory & { score: number; rerankScore?: number })[];
  try {
    docs = await memories
      .aggregate<Memory & { score: number; rerankScore?: number }>(pipeline)
      .toArray();
  } catch (err) {
    // Downgrade once, permanently, and retry. The demo should not die over
    // which tier of a stage the cluster happens to support.
    if (useNative && unsupportedStage(err, "rerank")) {
      nativeRerankDowngraded = true;
      // Print the server's own message. "Unrecognized stage" means the cluster
      // lacks the feature; anything else means our stage spec is wrong, and
      // silently downgrading would hide a bug we could actually fix.
      console.warn(
        `  ($rerank rejected, falling back to client-side reranking)\n  reason: ${(err as Error).message}`,
      );
      return searchSlice(query, queryVector, spec, tags);
    }
    if (useHybrid && unsupportedStage(err, "rankfusion")) {
      hybridDowngraded = true;
      console.warn(
        "  ($rankFusion unavailable — needs MongoDB 8.1+. Falling back to vector-only.)",
      );
      return searchSlice(query, queryVector, spec, tags);
    }
    throw err;
  }

  return docs.map((d) => ({
    memoryId: d.memoryId,
    kind: d.kind,
    text: d.text,
    score: d.score,
    rerankScore: d.rerankScore,
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

  // In auto mode we never embed anything — MongoDB vectorizes the query string
  // inside the database.
  const queryVector =
    config.embeddingMode === "client" ? await embedOne(query, "query") : null;

  const specs: SliceSpec[] = [
    { kind: "policy", customerId: null, limit: policyLimit },
    { kind: "episodic", customerId, limit: episodicLimit },
    { kind: "lesson", customerId: null, limit: lessonLimit },
  ];

  const slices = await Promise.all(
    specs.map((spec) =>
      searchSlice(query, queryVector, spec, spec.kind === "episodic" ? [] : tags),
    ),
  );

  // Native reranking already trimmed each slice to its quota inside the
  // pipeline. Otherwise rerank client-side, then trim.
  const finished = rerankPath() === "native";
  const selected: RecalledMemory[] = [];

  for (const [i, slice] of slices.entries()) {
    const limit = specs[i].limit;
    let ordered = slice;

    if (!finished && rerankPath() === "client") {
      const hits = await rerank(query, slice.map((m) => m.text));
      if (hits) {
        for (const hit of hits) slice[hit.index].rerankScore = hit.relevanceScore;
        ordered = hits.map((h) => slice[h.index]);
      }
    }

    // A lesson that has lost more than it won is actively misleading — drop it
    // and let reflection replace it.
    const kept = ordered.filter(
      (m) => !(m.kind === "lesson" && m.winRate !== undefined && m.winRate < 0.4),
    );
    selected.push(...kept.slice(0, limit));
  }

  return selected;
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
  const now = new Date();

  // undefined in auto mode: the document is written with no `embedding` field
  // and MongoDB vectorizes `text` on write.
  const embedding = await maybeEmbed(input.text);

  const doc: Memory = {
    memoryId: randomUUID(),
    kind: input.kind,
    customerId: input.customerId,
    scope: input.customerId ?? "global",
    text: input.text,
    ...(embedding ? { embedding } : {}),
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
