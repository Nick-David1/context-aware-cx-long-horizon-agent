import "dotenv/config";
import { MongoClient } from "mongodb";
import { config, assertConfig } from "../src/lib/config";

/**
 * Creates the Atlas Vector Search index for the context engine, plus the regular
 * indexes the dashboard queries rely on.
 *
 * Requires an Atlas cluster (M0 works). Vector Search does not exist on a local
 * mongod — if you point MONGODB_URI at localhost this will fail.
 */

const filterFields = [
  // Pre-filters. Without these the three-slice retrieval in memory.ts can't
  // separate policy from episodic from lesson, and superseded memories would
  // still be candidates.
  { type: "filter", path: "kind" },
  // `scope`, not `customerId` — see the note on Memory.scope in types.ts.
  { type: "filter", path: "scope" },
  { type: "filter", path: "tags" },
  { type: "filter", path: "active" },
];

/**
 * The lexical half of hybrid retrieval. `dynamic: false` so only the fields we
 * name are indexed; token types on the filter fields so they can be used in
 * compound filters rather than scored.
 */
const textDefinition = {
  mappings: {
    dynamic: false,
    fields: {
      [config.embedTextField]: { type: "string" },
      kind: { type: "token" },
      scope: { type: "token" },
      tags: [{ type: "string" }, { type: "token" }],
      active: { type: "boolean" },
    },
  },
};

/**
 * Auto mode: MongoDB generates the embeddings itself from the text field, using
 * a Voyage model — at index time as documents are written, and again at query
 * time for the query string. No vector ever crosses the wire from us, and no
 * model API key is needed in .env.
 */
const autoVectorDefinition = {
  fields: [
    {
      type: "autoEmbed",
      modality: "text",
      path: config.embedTextField,
      model: config.embedModel,
    },
    ...filterFields,
  ],
};

/** Client mode: we supply the vector, so the index declares its shape. */
const clientVectorDefinition = {
  fields: [
    {
      type: "vector",
      path: "embedding",
      numDimensions: config.embeddingDims,
      similarity: "cosine",
    },
    ...filterFields,
  ],
};

const vectorDefinition =
  config.embeddingMode === "auto" ? autoVectorDefinition : clientVectorDefinition;

/**
 * An index built before `scope` existed would silently fail to scope-filter, so
 * detect the drift and update in place rather than relying on someone
 * remembering to drop and recreate.
 */
function hasScopeFilter(idx: Record<string, unknown> | undefined): boolean {
  const def = idx?.latestDefinition as { fields?: { path?: string }[] } | undefined;
  return Boolean(def?.fields?.some((f) => f.path === "scope"));
}

/** Auto and client indexes differ in *type*, which updateSearchIndex can't convert. */
function modeOf(idx: Record<string, unknown> | undefined): "auto" | "client" | null {
  const def = idx?.latestDefinition as { fields?: { type?: string }[] } | undefined;
  if (!def?.fields) return null;
  if (def.fields.some((f) => f.type === "autoEmbed")) return "auto";
  if (def.fields.some((f) => f.type === "vector")) return "client";
  return null;
}

async function main() {
  assertConfig();
  const client = await new MongoClient(config.mongoUri).connect();
  const db = client.db(config.dbName);

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
    db.collection("memories").createIndex({ scope: 1, kind: 1, active: 1 }),
    db.collection("outcomes").createIndex({ caseId: 1 }),
    // One live checkpoint per case — resume looks it up by this pair.
    db.collection("checkpoints").createIndex({ caseId: 1 }, { unique: true }),
    db.collection("checkpoints").createIndex({ status: 1, updatedAt: 1 }),
  ]);
  console.log("regular indexes ready");

  const memories = db.collection("memories");
  const existingSearch = (await memories.listSearchIndexes().toArray()) as Record<
    string,
    unknown
  >[];
  const vectorIdx = existingSearch.find((i) => i.name === config.vectorIndex);
  const existingMode = modeOf(vectorIdx);

  const label =
    config.embeddingMode === "auto"
      ? `auto-embed in Atlas via ${config.embedModel}`
      : `${config.embeddingDims} dims, client-supplied`;

  if (!vectorIdx) {
    await memories.createSearchIndex({
      name: config.vectorIndex,
      type: "vectorSearch",
      definition: vectorDefinition,
    });
    console.log(`created vector index ${config.vectorIndex} (${label})`);
  } else if (existingMode && existingMode !== config.embeddingMode) {
    // Switching between auto and client changes the index type, not just its
    // fields. Rebuild rather than failing cryptically at query time.
    console.log(
      `vector index is ${existingMode}-mode but EMBEDDING_MODE=${config.embeddingMode}`,
    );
    console.log("  dropping and recreating — the index type cannot be updated in place");
    await memories.dropSearchIndex(config.vectorIndex);
    await new Promise((r) => setTimeout(r, 3000)); // Atlas needs the name freed
    await memories.createSearchIndex({
      name: config.vectorIndex,
      type: "vectorSearch",
      definition: vectorDefinition,
    });
    console.log(`  recreated as ${label}`);
    console.log("  NOTE: re-run `npm run db:seed` — documents need re-writing for the new mode");
  } else if (!hasScopeFilter(vectorIdx)) {
    await memories.updateSearchIndex(config.vectorIndex, vectorDefinition);
    console.log(`updated vector index ${config.vectorIndex} — added the 'scope' filter`);
  } else {
    console.log(`vector index ${config.vectorIndex} is current (${label})`);
  }

  const textIdx = existingSearch.find((i) => i.name === config.textIndex);
  if (!textIdx) {
    await memories.createSearchIndex({
      name: config.textIndex,
      type: "search",
      definition: textDefinition,
    });
    console.log(`created search index ${config.textIndex} (BM25, for hybrid retrieval)`);
  } else {
    await memories.updateSearchIndex(config.textIndex, textDefinition);
    console.log(`search index ${config.textIndex} updated`);
  }

  console.log("\nAtlas builds search indexes asynchronously — they usually go READY");
  console.log("within ~1 minute. Run `npm run verify` to check status.");

  await client.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
