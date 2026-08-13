import "dotenv/config";
import { MongoClient } from "mongodb";
import Anthropic from "@anthropic-ai/sdk";
import { config } from "../src/lib/config";
import { embed, rerank, EMBEDDING_DIM } from "../src/lib/embeddings";

/**
 * Preflight check for every external dependency.
 *
 * Run before demoing. Each check is independent, so a failure names the exact
 * credential or resource that is wrong instead of surfacing as a confusing
 * runtime error three layers deep.
 */

type Result = { name: string; ok: boolean; required: boolean; detail: string };
const results: Result[] = [];

async function check(name: string, required: boolean, fn: () => Promise<string>) {
  try {
    results.push({ name, required, ok: true, detail: await fn() });
  } catch (err) {
    results.push({
      name,
      required,
      ok: false,
      detail: err instanceof Error ? err.message : String(err),
    });
  }
}

console.log(
  `\nmode: retrieval=${config.retrievalMode}  embedding=${config.embeddingMode}  ` +
    `rerank=${config.rerankMode}  model=${config.embedModel}`,
);

await check("MongoDB Atlas — connection", true, async () => {
  if (!config.mongoUri) throw new Error("MONGODB_URI not set");
  if (!config.mongoUri.includes("mongodb+srv")) {
    throw new Error("Not an Atlas SRV string — Vector Search needs Atlas, not local mongod");
  }
  const client = await new MongoClient(config.mongoUri, {
    serverSelectionTimeoutMS: 8000,
  }).connect();
  const db = client.db(config.dbName);
  const counts = await Promise.all(
    ["customers", "loans", "cases", "memories"].map(
      async (c) => `${c}=${await db.collection(c).countDocuments()}`,
    ),
  );
  await client.close();
  return counts.join(" ");
});

await check("MongoDB Atlas — vector index", true, async () => {
  if (!config.mongoUri) throw new Error("MONGODB_URI not set");
  const client = await new MongoClient(config.mongoUri, {
    serverSelectionTimeoutMS: 8000,
  }).connect();
  const db = client.db(config.dbName);
  // The driver types this as {name} only; the server also returns status,
  // queryable, and latestDefinition — all of which we need here.
  const indexes = (await db.collection("memories").listSearchIndexes().toArray()) as Record<
    string,
    unknown
  >[];
  await client.close();

  const idx = indexes.find((i) => i.name === config.vectorIndex);
  if (!idx) throw new Error(`${config.vectorIndex} missing — run \`npm run db:indexes\``);

  const def = idx.latestDefinition as { fields?: { type?: string }[] } | undefined;
  const mode = def?.fields?.some((f) => f.type === "autoEmbed") ? "auto" : "client";
  if (mode !== config.embeddingMode) {
    throw new Error(
      `index is ${mode}-mode but EMBEDDING_MODE=${config.embeddingMode} — re-run \`npm run db:indexes\``,
    );
  }

  const status = String(idx.status ?? "unknown");
  if (!idx.queryable) throw new Error(`status ${status} — still building, wait and re-run`);
  return `${config.vectorIndex} ${status}, ${mode}-mode`;
});

await check("MongoDB Atlas — text index", config.retrievalMode === "hybrid", async () => {
  if (config.retrievalMode !== "hybrid") return "not used (RETRIEVAL_MODE=vector)";
  const client = await new MongoClient(config.mongoUri, {
    serverSelectionTimeoutMS: 8000,
  }).connect();
  const indexes = (await client
    .db(config.dbName)
    .collection("memories")
    .listSearchIndexes()
    .toArray()) as Record<string, unknown>[];
  await client.close();

  const idx = indexes.find((i) => i.name === config.textIndex);
  if (!idx) throw new Error(`${config.textIndex} missing — run \`npm run db:indexes\``);
  const status = String(idx.status ?? "unknown");
  if (!idx.queryable) throw new Error(`status ${status} — still building, wait and re-run`);
  return `${config.textIndex} ${status}, BM25 for $rankFusion`;
});

// Only required in client mode — in auto mode Atlas holds the model key.
await check("Embeddings", config.embeddingMode === "client", async () => {
  if (config.embeddingMode === "auto") {
    return "handled by Atlas in-database (no key needed here)";
  }
  if (!config.voyageKey) throw new Error("VOYAGE_API_KEY not set");
  const [v] = await embed(["payment date change eligibility"], "query");
  if (v.length !== EMBEDDING_DIM) {
    throw new Error(`got ${v.length} dims, expected ${EMBEDDING_DIM} — model/index mismatch`);
  }
  return `${config.embedModel}, ${v.length} dims, via ${config.voyageBaseUrl}`;
});

// Native reranking is validated by the first real query, not here — this check
// only covers the client-side path.
await check("Reranking", false, async () => {
  if (config.rerankMode === "off") return "disabled";
  if (config.rerankMode === "native") {
    return `native $rerank (${config.rerankModel}) — needs MongoDB 8.3 + Native Reranking on; falls back automatically`;
  }
  const hits = await rerank("can I move my payment date?", [
    "Payment date changes are allowed once per 90 days on loans that are current.",
    "Photosynthesis converts light energy into glucose.",
  ]);
  if (!hits) throw new Error("client rerank returned nothing — check VOYAGE_API_KEY");
  if (hits[0].index !== 0) throw new Error("reranker ranked the irrelevant doc first");
  return `${config.rerankModel} client-side, top score ${hits[0].relevanceScore.toFixed(3)}`;
});

await check("Anthropic", true, async () => {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY not set");
  const res = await new Anthropic().messages.create({
    model: config.agentModel,
    max_tokens: 16,
    messages: [{ role: "user", content: "Reply with the single word: ready" }],
  });
  const text = res.content.find((b) => b.type === "text");
  return `${config.agentModel} → "${text && "text" in text ? text.text.trim() : ""}"`;
});

await check("ElevenLabs", false, async () => {
  const key = process.env.ELEVENLABS_API_KEY;
  const agentId = process.env.ELEVENLABS_AGENT_ID;
  if (!key) throw new Error("ELEVENLABS_API_KEY not set (voice disabled)");
  if (!agentId) throw new Error("ELEVENLABS_AGENT_ID not set (voice disabled)");

  const res = await fetch(`https://api.elevenlabs.io/v1/convai/agents/${agentId}`, {
    headers: { "xi-api-key": key },
  });
  if (!res.ok) throw new Error(`agent lookup failed (${res.status}): ${await res.text()}`);
  const agent = (await res.json()) as { name?: string };
  return `agent "${agent.name ?? agentId}" reachable`;
});

console.log("");
for (const r of results) {
  const mark = r.ok ? "✓" : r.required ? "✗" : "–";
  console.log(`${mark}  ${r.name.padEnd(30)} ${r.detail}`);
}

const failed = results.filter((r) => r.required && !r.ok);
console.log(
  failed.length === 0
    ? "\nAll required checks passed.\n"
    : `\n${failed.length} required check(s) failed — see above.\n`,
);
process.exit(failed.length === 0 ? 0 : 1);
