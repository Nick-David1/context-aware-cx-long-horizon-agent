import { MongoClient, type Db, type Collection } from "mongodb";
import type {
  Checkpoint,
  Customer,
  HorizonCase,
  Interaction,
  Loan,
  Memory,
  Outcome,
} from "./types";

// Next dev server hot-reloads modules; cache the client on globalThis so we don't
// open a new connection pool on every reload. Connect lazily — a module-scope
// connect() fires during `next build` and fails the page-data collection pass.
const globalForMongo = globalThis as unknown as { _horizonMongo?: Promise<MongoClient> };

function connect(): Promise<MongoClient> {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    throw new Error("MONGODB_URI is not set — copy .env.example to .env and fill it in.");
  }
  const promise = new MongoClient(uri).connect();
  globalForMongo._horizonMongo = promise;
  return promise;
}

export async function getDb(): Promise<Db> {
  const client = await (globalForMongo._horizonMongo ?? connect());
  return client.db(process.env.MONGODB_DB ?? "horizon");
}

export async function collections(): Promise<{
  customers: Collection<Customer>;
  loans: Collection<Loan>;
  cases: Collection<HorizonCase>;
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
    cases: db.collection<HorizonCase>("cases"),
    interactions: db.collection<Interaction>("interactions"),
    memories: db.collection<Memory>("memories"),
    outcomes: db.collection<Outcome>("outcomes"),
  };
}

export const VECTOR_INDEX = "memory_vector_index";
