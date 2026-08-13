import { MongoClient, type Db, type Collection } from "mongodb";
import { config } from "./config";
import type {
  Checkpoint,
  Customer,
  CaseRecord,
  Interaction,
  Loan,
  Memory,
  Outcome,
} from "./types";

// Next dev server hot-reloads modules; cache the client on globalThis so we don't
// open a new connection pool on every reload. Connect lazily — a module-scope
// connect() fires during `next build` and fails the page-data collection pass.
const globalForMongo = globalThis as unknown as { _cxMongo?: Promise<MongoClient> };

function connect(): Promise<MongoClient> {
  if (!config.mongoUri) {
    throw new Error("MONGODB_URI is not set — copy .env.example to .env and fill it in.");
  }
  const promise = new MongoClient(config.mongoUri).connect();
  globalForMongo._cxMongo = promise;
  return promise;
}

export async function getDb(): Promise<Db> {
  const client = await (globalForMongo._cxMongo ?? connect());
  return client.db(config.dbName);
}

export async function collections(): Promise<{
  customers: Collection<Customer>;
  loans: Collection<Loan>;
  cases: Collection<CaseRecord>;
  interactions: Collection<Interaction>;
  memories: Collection<Memory>;
  outcomes: Collection<Outcome>;
  checkpoints: Collection<Checkpoint>;
}> {
  const db = await getDb();
  return {
    checkpoints: db.collection<Checkpoint>("checkpoints"),
    customers: db.collection<Customer>("customers"),
    loans: db.collection<Loan>("loans"),
    cases: db.collection<CaseRecord>("cases"),
    interactions: db.collection<Interaction>("interactions"),
    memories: db.collection<Memory>("memories"),
    outcomes: db.collection<Outcome>("outcomes"),
  };
}

export const VECTOR_INDEX = config.vectorIndex;
