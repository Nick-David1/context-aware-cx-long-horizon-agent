import { NextResponse } from "next/server";
import { collections } from "@/lib/mongo";
import { openCase } from "@/lib/actions";
import type { CaseGoal } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const { cases, customers, loans } = await collections();
  const all = await cases.find({}).sort({ updatedAt: -1 }).limit(50).toArray();

  const customerIds = [...new Set(all.map((c) => c.customerId))];
  const [people, allLoans] = await Promise.all([
    customers.find({ customerId: { $in: customerIds } }).toArray(),
    loans.find({ customerId: { $in: customerIds } }).toArray(),
  ]);
  const byId = new Map(people.map((p) => [p.customerId, p]));

  return NextResponse.json({
    cases: all.map((c) => ({
      ...c,
      _id: undefined,
      customerName: byId.get(c.customerId)?.name ?? c.customerId,
      loan: allLoans.find((l) => l.loanId === c.loanId) ?? null,
    })),
  });
}

export async function POST(req: Request) {
  const body = (await req.json()) as {
    customerId: string;
    goal: CaseGoal;
    loanId?: string;
  };
  const kase = await openCase(body.customerId, body.goal, body.loanId);
  return NextResponse.json({ caseId: kase.caseId });
}
