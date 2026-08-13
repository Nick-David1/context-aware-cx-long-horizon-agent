import { NextResponse } from "next/server";
import { collections } from "@/lib/mongo";
import { recall } from "@/lib/memory";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Browse the context engine, or run a live vector search against it. */
export async function GET(req: Request) {
  const url = new URL(req.url);
  const query = url.searchParams.get("q");
  const customerId = url.searchParams.get("customerId");

  if (query && customerId) {
    return NextResponse.json({ results: await recall({ query, customerId }) });
  }

  const { memories } = await collections();
  const all = await memories
    .find({ active: true }, { projection: { embedding: 0 } })
    .sort({ createdAt: -1 })
    .limit(100)
    .toArray();

  return NextResponse.json({ memories: all });
}
