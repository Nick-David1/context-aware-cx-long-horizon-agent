/**
 * Embeddings for Atlas Vector Search.
 *
 * Primary path is Voyage AI (voyage-3, 1024 dims). If VOYAGE_API_KEY is unset we
 * fall back to a deterministic local hashing embedder of the same dimensionality
 * so the whole demo still runs without a third API key. The fallback retrieves
 * far worse — set VOYAGE_API_KEY before demoing.
 */

export const EMBEDDING_DIM = 1024;

const VOYAGE_MODEL = "voyage-3";

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
    body: JSON.stringify({ input: texts, model: VOYAGE_MODEL, input_type: inputType }),
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
