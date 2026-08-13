/**
 * Embeddings and reranking for Atlas Vector Search.
 *
 * Uses MongoDB's native Voyage AI models, keyed through an Atlas Model API key
 * (Atlas UI → AI Model APIs → Create model API key). Same endpoint as standalone
 * Voyage; the key is issued and billed by Atlas, so the whole retrieval stack —
 * store, index, embed, rerank — sits behind one vendor.
 *
 * If VOYAGE_API_KEY is unset we fall back to a deterministic local hashing
 * embedder of the same dimensionality so the app still runs. The fallback
 * retrieves far worse and cannot rerank at all.
 */

export const EMBEDDING_DIM = 1024;

const EMBED_MODEL = "voyage-4-large";
const RERANK_MODEL = "rerank-2.5";

export function usingRealEmbeddings(): boolean {
  return Boolean(process.env.VOYAGE_API_KEY);
}

export async function embed(
  texts: string[],
  inputType: "document" | "query",
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const key = process.env.VOYAGE_API_KEY;
  if (!key) return texts.map(hashEmbed);

  const res = await fetch("https://api.voyageai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({ input: texts, model: EMBED_MODEL, input_type: inputType }),
  });

  if (!res.ok) {
    throw new Error(`Voyage embeddings failed (${res.status}): ${await res.text()}`);
  }

  const json = (await res.json()) as { data: { index: number; embedding: number[] }[] };
  // Voyage returns results with an explicit index; don't assume array order.
  const out = new Array<number[]>(texts.length);
  for (const row of json.data) out[row.index] = row.embedding;
  return out;
}

export interface RerankHit {
  /** Index into the `documents` array that was passed in. */
  index: number;
  relevanceScore: number;
}

/**
 * Reranks candidates against the query with a cross-encoder.
 *
 * Vector search is a bi-encoder: query and document are embedded separately, so
 * it is fast over millions of docs but blunt at the top of the list. The
 * reranker scores each (query, document) pair jointly, which is far more
 * accurate but too slow to run over the whole collection. Standard pattern is
 * what we do here: over-fetch with $vectorSearch, then rerank the shortlist.
 *
 * Returns null when no key is configured, so callers fall back to vector order.
 */
export async function rerank(
  query: string,
  documents: string[],
): Promise<RerankHit[] | null> {
  const key = process.env.VOYAGE_API_KEY;
  if (!key || documents.length === 0) return null;

  const res = await fetch("https://api.voyageai.com/v1/rerank", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify({ query, documents, model: RERANK_MODEL }),
  });

  if (!res.ok) {
    // Retrieval still works without reranking — degrade instead of failing the turn.
    console.error(`Voyage rerank failed (${res.status}): ${await res.text()}`);
    return null;
  }

  const json = (await res.json()) as {
    data: { index: number; relevance_score: number }[];
  };
  return json.data.map((d) => ({ index: d.index, relevanceScore: d.relevance_score }));
}

export async function embedOne(
  text: string,
  inputType: "document" | "query",
): Promise<number[]> {
  const [v] = await embed([text], inputType);
  return v;
}

/** Deterministic bag-of-words hash embedding, L2-normalized. Demo fallback only. */
function hashEmbed(text: string): number[] {
  const vec = new Array<number>(EMBEDDING_DIM).fill(0);
  const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  for (const token of tokens) {
    let h = 2166136261;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const idx = Math.abs(h) % EMBEDDING_DIM;
    vec[idx] += 1;
  }
  const norm = Math.hypot(...vec) || 1;
  return vec.map((v) => v / norm);
}
