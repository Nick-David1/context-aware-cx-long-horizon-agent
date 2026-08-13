import { config } from "./config";

/**
 * Embedding and reranking, for the paths where *we* are the ones doing it.
 *
 * In the default `auto` embedding mode this file is barely used: MongoDB
 * generates embeddings inside the database from the indexed text field, at
 * write time and at query time, so no vector ever crosses the wire. These
 * functions exist for `client` mode and for client-side reranking.
 *
 * MongoDB acquired Voyage AI in Feb 2025, so these are MongoDB's own retrieval
 * models exposed natively through Atlas — one vendor for store, index, embed,
 * and rerank. See config.ts for the two-base-URL trap.
 */

export const EMBEDDING_DIM = config.embeddingDims;

export function usingRealEmbeddings(): boolean {
  return config.embeddingMode === "auto" || Boolean(config.voyageKey);
}

/**
 * Whether we must produce a vector ourselves. False in auto mode, where the
 * document is written with no `embedding` field and MongoDB vectorizes it.
 */
export function clientEmbeddingRequired(): boolean {
  return config.embeddingMode === "client";
}

export async function embed(
  texts: string[],
  inputType: "document" | "query",
): Promise<number[][]> {
  if (texts.length === 0) return [];
  if (!config.voyageKey) return texts.map(hashEmbed);

  const res = await fetch(`${config.voyageBaseUrl}/embeddings`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${config.voyageKey}`,
    },
    body: JSON.stringify({ input: texts, model: config.embedModel, input_type: inputType }),
  });

  if (!res.ok) {
    throw new Error(
      `Embeddings failed (${res.status}) at ${config.voyageBaseUrl}: ${await res.text()}`,
    );
  }

  const json = (await res.json()) as { data: { embedding: number[]; index: number }[] };
  // Results carry an explicit index; don't assume array order.
  return json.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
}

export async function embedOne(
  text: string,
  inputType: "document" | "query",
): Promise<number[]> {
  const [v] = await embed([text], inputType);
  return v;
}

/** Embed only if we're the ones responsible. Returns undefined in auto mode. */
export async function maybeEmbed(text: string): Promise<number[] | undefined> {
  if (!clientEmbeddingRequired()) return undefined;
  return embedOne(text, "document");
}

export interface RerankHit {
  /** Index into the `documents` array that was passed in. */
  index: number;
  relevanceScore: number;
}

/**
 * Client-side cross-encoder reranking, used when `$rerank` isn't available.
 *
 * Deliberately fail-open: if the key is missing or the call errors, retrieval
 * degrades to vector order rather than the turn failing. A slightly worse
 * answer beats a 500 on stage.
 */
export async function rerank(
  query: string,
  documents: string[],
): Promise<RerankHit[] | null> {
  if (!config.voyageKey || documents.length === 0) return null;

  try {
    const res = await fetch(`${config.voyageBaseUrl}/rerank`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.voyageKey}`,
      },
      body: JSON.stringify({
        query,
        documents,
        model: config.rerankModel,
        top_k: documents.length,
        truncation: true,
      }),
    });

    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);

    // ai.mongodb.com returns the rows under `data`; the Voyage docs and the
    // standalone api.voyageai.com surface document `results`. Accept either —
    // this is the exact mismatch that made reranking silently no-op.
    const json = (await res.json()) as {
      results?: { index: number; relevance_score: number }[];
      data?: { index: number; relevance_score: number }[];
    };
    const rows = json.results ?? json.data;
    if (!rows) {
      throw new Error(`no results/data array; response keys were: ${Object.keys(json).join(", ")}`);
    }
    return rows.map((r) => ({ index: r.index, relevanceScore: r.relevance_score }));
  } catch (err) {
    console.warn(`  (rerank failed, using vector order: ${(err as Error).message})`);
    return null;
  }
}

/** Deterministic bag-of-words hash embedding, L2-normalized. Last-resort fallback. */
function hashEmbed(text: string): number[] {
  const vec = new Array<number>(EMBEDDING_DIM).fill(0);
  const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  for (const token of tokens) {
    let h = 2166136261;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    vec[Math.abs(h) % EMBEDDING_DIM] += 1;
  }
  const norm = Math.hypot(...vec) || 1;
  return vec.map((v) => v / norm);
}
