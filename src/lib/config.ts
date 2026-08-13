/**
 * Config never throws at import time — otherwise any `npm run` before .env
 * exists dumps a stack trace instead of a useful message. Missing values are
 * collected and reported by whichever component actually needs them.
 */

/**
 * Who generates the embeddings.
 *
 *   auto   — MongoDB does it, inside the database, using Voyage models. The
 *            vector index declares `type: "autoEmbed"` over a text field, and
 *            $vectorSearch takes a plain `query` string instead of a
 *            `queryVector`. No embedding call in our code, at write or at read.
 *
 *   client — we call the embedding API ourselves and store the vector. The
 *            portable path, and the fallback if auto-embedding is unavailable.
 *
 * `npm run verify` probes both and reports which your cluster supports.
 */
export type EmbeddingMode = "auto" | "client";

/**
 * Where reranking happens.
 *
 *   native — MongoDB's `$rerank` aggregation stage. Runs inside the query
 *            pipeline and needs no API key of ours. Requires MongoDB 8.3+ with
 *            Native Reranking enabled in Atlas Project Settings.
 *   client — we POST to the embedding/reranking API ourselves. Needs a key.
 *   off    — vector order only.
 *
 * In native mode, retrieval downgrades to client automatically the first time
 * the stage is rejected, so an older cluster still works.
 */
export type RerankMode = "native" | "client" | "off";

export const config = {
  mongoUri: process.env.MONGODB_URI ?? "",
  dbName: process.env.MONGODB_DB ?? "cx_agent",
  vectorIndex: "memory_vector_index",

  embeddingMode: (process.env.EMBEDDING_MODE ?? "auto") as EmbeddingMode,
  /** The field autoEmbed indexes, and the field we embed in client mode. */
  embedTextField: "text",
  /** Only meaningful in client mode — in auto mode MongoDB owns the dimensions. */
  embeddingDims: 1024,

  /**
   * There are TWO surfaces for these models and a key works against exactly one:
   *
   *   https://ai.mongodb.com/v1    — keys from the Atlas UI (AI Model APIs)
   *   https://api.voyageai.com/v1  — keys from voyageai.com directly
   *
   * Using a key against the wrong base URL is a 403 whose message does say so,
   * but only if you read it. Default to the Atlas surface.
   */
  voyageKey: process.env.VOYAGE_API_KEY ?? "",
  voyageBaseUrl: (process.env.VOYAGE_BASE_URL ?? "https://ai.mongodb.com/v1").replace(/\/$/, ""),
  /**
   * voyage-4 is the current general-purpose recommendation. voyage-4-large,
   * voyage-4-lite, and voyage-3-large are all 1024 dims too, so switching
   * between them does not require rebuilding the index.
   */
  embedModel: process.env.VOYAGE_EMBED_MODEL ?? "voyage-4",

  rerankMode: (process.env.RERANK_MODE ?? "native") as RerankMode,
  rerankModel: process.env.RERANK_MODEL ?? "rerank-2.5",

  agentModel: process.env.AGENT_MODEL ?? "claude-opus-5",
  reflectionModel: process.env.REFLECTION_MODEL ?? "claude-opus-5",
};

/**
 * Call at the top of any command that talks to a service.
 *
 * In auto mode the model key lives on the Atlas cluster, configured in the
 * Atlas UI — our process never calls an embedding API, so VOYAGE_API_KEY is
 * genuinely not required in .env. It is only needed for client-side embedding
 * or client-side reranking.
 */
export function assertConfig(): void {
  const missing: string[] = [];
  if (!config.mongoUri) missing.push("MONGODB_URI");

  if (config.embeddingMode === "client" && !config.voyageKey) {
    missing.push("VOYAGE_API_KEY (EMBEDDING_MODE=client needs it)");
  }
  if (config.rerankMode === "client" && !config.voyageKey) {
    missing.push("VOYAGE_API_KEY (RERANK_MODE=client needs it)");
  }

  if (missing.length) {
    throw new Error(
      `Missing required env var(s): ${missing.join(", ")}\n` +
        "Copy .env.example to .env and fill them in, then run `npm run verify`.",
    );
  }
}
