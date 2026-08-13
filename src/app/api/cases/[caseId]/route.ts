import { NextResponse } from "next/server";
import { collections } from "@/lib/mongo";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ caseId: string }> },
) {
  const { caseId } = await params;
  const { cases, customers, loans, interactions, memories } = await collections();

  const kase = await cases.findOne({ caseId });
  if (!kase) return NextResponse.json({ error: "not found" }, { status: 404 });

  const [customer, customerLoans, history, customerMemories] = await Promise.all([
    customers.findOne({ customerId: kase.customerId }),
    loans.find({ customerId: kase.customerId }).toArray(),
    interactions.find({ caseId }).sort({ startedAt: 1 }).toArray(),
    memories
      .find(
        { customerId: kase.customerId, active: true },
        { projection: { embedding: 0 } },
      )
      .sort({ createdAt: -1 })
      .limit(20)
      .toArray(),
  ]);

  return NextResponse.json({
    case: kase,
    customer,
    loans: customerLoans,
    interactions: history,
    memories: customerMemories,
  });
}
