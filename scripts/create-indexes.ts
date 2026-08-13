import "dotenv/config";
import { MongoClient } from "mongodb";
import { EMBEDDING_DIM } from "../src/lib/embeddings";

/**
 * Creates the Atlas Vector Search index for the context engine, plus the regular
 * indexes the dashboard queries rely on.
 *
 * Requires an Atlas cluster (M0 works). Vector Search does not exist on a local
 * mongod — if you point MONGODB_URI at localhost this will fail.
 */

const VECTOR_INDEX = "memory_vector_index";

async function main() {
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error("MONGODB_URI is not set");
  const client = await new MongoClient(uri).connect();
  const db = client.db(process.env.MONGODB_DB ?? "cx_agent");

  const collectionNames = [
    "customers",
    "loans",
    "cases",
    "interactions",
    "memories",
    "outcomes",
    "checkpoints",
  ];
  for (const name of collectionNames) {
    const existing = await db.listCollections({ name }).toArray();
    if (existing.length === 0) {
      await db.createCollection(name);
      console.log(`created collection ${name}`);
    }
  }

  await Promise.all([
    db.collection("customers").createIndex({ customerId: 1 }, { unique: true }),
    db.collection("loans").createIndex({ loanId: 1 }, { unique: true }),
    db.collection("loans").createIndex({ customerId: 1 }),
    db.collection("cases").createIndex({ caseId: 1 }, { unique: true }),
    // Drives the follow-up tick: find open cases whose next action is due.
    db.collection("cases").createIndex({ status: 1, nextActionAt: 1 }),
    db.collection("interactions").createIndex({ caseId: 1, startedAt: 1 }),
    db.collection("memories").createIndex({ memoryId: 1 }, { unique: true }),
    db.collection("outcomes").createIndex({ caseId: 1 }),
    // One live checkpoint per case — resume looks it up by this pair.
    db.collection("checkpoints").createIndex({ caseId: 1 }, { unique: true }),
    db.collection("checkpoints").createIndex({ status: 1, updatedAt: 1 }),
  ]);
  console.log("regular indexes ready");

  const memories = db.collection("memories");
  const existingSearch = await memories.listSearchIndexes().toArray();

  if (existingSearch.some((i) => i.name === VECTOR_INDEX)) {
    console.log(`vector index "${VECTOR_INDEX}" already exists — skipping`);
  } else {
    await memories.createSearchIndex({
      name: VECTOR_INDEX,
      type: "vectorSearch",
      definition: {
        fields: [
          { type: "vector", path: "embedding", numDimensions: EMBEDDING_DIM, similarity: "cosine" },
          // Pre-filters. Without these the three-slice retrieval in memory.ts
          // can't separate policy from episodic from lesson.
          { type: "filter", path: "kind" },
          { type: "filter", path: "customerId" },
          { type: "filter", path: "tags" },
          { type: "filter", path: "active" },
        ],
      },
    });
    console.log(`vector index "${VECTOR_INDEX}" created — Atlas takes ~1 minute to build it`);
  }

  await client.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
