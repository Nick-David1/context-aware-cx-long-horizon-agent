import { NextResponse } from "next/server";
import { collections } from "@/lib/mongo";
import { startOutboundCall } from "@/lib/elevenlabs";

export const runtime = "nodejs";
export const maxDuration = 60;

/** Manually place the outbound call for a case — the "call me now" demo button. */
export async function POST(req: Request) {
  const { caseId } = (await req.json()) as { caseId: string };
  const { cases, customers } = await collections();

  const kase = await cases.findOne({ caseId });
  if (!kase) return NextResponse.json({ error: "case not found" }, { status: 404 });

  const customer = await customers.findOne({ customerId: kase.customerId });
  if (!customer) return NextResponse.json({ error: "customer not found" }, { status: 404 });

  try {
    const call = await startOutboundCall({
      toNumber: customer.phone,
      variables: { caseId, customerName: customer.name },
    });
    return NextResponse.json({ placed: true, call });
  } catch (err) {
    return NextResponse.json(
      { placed: false, error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
