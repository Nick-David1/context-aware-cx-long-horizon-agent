import "dotenv/config";
import { MongoClient } from "mongodb";
import Anthropic from "@anthropic-ai/sdk";
import { embed, rerank, EMBEDDING_DIM } from "../src/lib/embeddings";

/**
 * Preflight check for every external dependency.
 *
 * Run this before demoing. Each check is independent, so a failure tells you
 * exactly which credential or resource is wrong instead of surfacing as a
 * confusing runtime error three layers deep.
 */

type Result = { name: string; ok: boolean; detail: string };
const results: Result[] = [];

async function check(name: string, fn: () => Promise<string>): Promise<void> {
  try {
    results.push({ name, ok: true, detail: await fn() });
  } catch (err) {
    results.push({ name, ok: false, detail: err instanceof Error ? err.message : String(err) });
  }
}

await check("MongoDB Atlas — connection", async () => {
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error("MONGODB_URI not set");
  if (!uri.includes("mongodb+srv")) {
    throw new Error("Not an Atlas SRV string — Vector Search needs Atlas, not local mongod");
  }
  const client = await new MongoClient(uri, { serverSelectionTimeoutMS: 8000 }).connect();
  const db = client.db(process.env.MONGODB_DB ?? "cx_agent");
  const counts = await Promise.all(
    ["customers", "loans", "cases", "memories"].map(async (c) => `${c}=${await db.collection(c).countDocuments()}`),
  );
  await client.close();
  return counts.join(" ");
});

await check("MongoDB Atlas — vector index", async () => {
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error("MONGODB_URI not set");
  const client = await new MongoClient(uri, { serverSelectionTimeoutMS: 8000 }).connect();
  const db = client.db(process.env.MONGODB_DB ?? "cx_agent");
  // The driver types listSearchIndexes() as {name} only; status is present on the
  // wire and is the field we actually need.
  const indexes = (await db
    .collection("memories")
    .listSearchIndexes()
    .toArray()) as { name: string; status?: string }[];
  await client.close();

  const idx = indexes.find((i) => i.name === "memory_vector_index");
  if (!idx) throw new Error("memory_vector_index missing — run `npm run db:indexes`");
  if (idx.status !== "READY") {
    throw new Error(`index status is ${idx.status}, not READY — wait for Atlas to finish building`);
  }
  return "memory_vector_index READY";
});

await check("Voyage (via Atlas) — embeddings", async () => {
  if (!process.env.VOYAGE_API_KEY) {
    throw new Error("VOYAGE_API_KEY not set — falling back to the local embedder");
  }
  const [v] = await embed(["payment date change eligibility"], "query");
  if (v.length !== EMBEDDING_DIM) {
    throw new Error(`got ${v.length} dims, expected ${EMBEDDING_DIM} — model/index mismatch`);
  }
  return `voyage-4-large, ${v.length} dims`;
});

await check("Voyage (via Atlas) — reranking", async () => {
  const hits = await rerank("can I move my payment date?", [
    "Payment date changes are allowed once per 90 days on loans that are current.",
    "Photosynthesis converts light energy into glucose.",
  ]);
  if (!hits) throw new Error("no key configured — reranking disabled");
  if (hits[0].index !== 0) throw new Error("reranker ranked the irrelevant doc first");
  return `rerank-2.5, top score ${hits[0].relevanceScore.toFixed(3)}`;
});

await check("Anthropic", async () => {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY not set");
  const res = await new Anthropic().messages.create({
    model: "claude-opus-5",
    max_tokens: 16,
    messages: [{ role: "user", content: "Reply with the single word: ready" }],
  });
  const text = res.content.find((b) => b.type === "text");
  return `claude-opus-5 → "${text && "text" in text ? text.text.trim() : ""}"`;
});

await check("ElevenLabs", async () => {
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) throw new Error("ELEVENLABS_API_KEY not set (voice disabled)");
  const agentId = process.env.ELEVENLABS_AGENT_ID;
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
  console.log(`${r.ok ? "✓" : "✗"}  ${r.name.padEnd(34)} ${r.detail}`);
}

const required = results.slice(0, 5); // ElevenLabs is optional for the text demo
const failed = required.filter((r) => !r.ok);
console.log(
  failed.length === 0
    ? "\nAll required checks passed.\n"
    : `\n${failed.length} required check(s) failed — see above.\n`,
);
process.exit(failed.length === 0 ? 0 : 1);
