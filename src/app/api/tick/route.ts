import { NextResponse } from "next/server";
import { collections } from "@/lib/mongo";
import { startOutboundCall } from "@/lib/elevenlabs";

export const runtime = "nodejs";
export const maxDuration = 120;

/**
 * The long-horizon heartbeat.
 *
 * Finds every case whose scheduled follow-up has come due and re-engages the
 * customer — by phone if voice is configured, otherwise just reporting what it
 * would do. Run it on a cron (Vercel Cron, GitHub Actions, or `npm run tick`).
 */
export async function POST(req: Request) {
  const url = new URL(req.url);
  const dryRun = url.searchParams.get("dryRun") === "true";
  // `upcoming` also lists follow-ups the agent has scheduled for later, so the
  // proactive-engagement story is visible without waiting for the clock.
  const upcoming = url.searchParams.get("upcoming") === "true";
  const { cases, customers } = await collections();

  const due = await cases
    .find({
      status: { $in: ["open", "waiting_on_customer"] },
      nextActionAt: upcoming ? { $exists: true } : { $lte: new Date() },
    })
    .limit(20)
    .toArray();

  const results = [];
  for (const kase of due) {
    const customer = await customers.findOne({ customerId: kase.customerId });
    if (!customer) continue;

    if (dryRun || !process.env.ELEVENLABS_AGENT_ID) {
      results.push({
        caseId: kase.caseId,
        customer: customer.name,
        wouldCall: customer.phone,
        reason: kase.nextActionReason,
        dueAt: kase.nextActionAt,
        due: (kase.nextActionAt?.getTime() ?? 0) <= Date.now(),
        placed: false,
      });
      continue;
    }

    try {
      const call = await startOutboundCall({
        toNumber: customer.phone,
        variables: { caseId: kase.caseId, customerName: customer.name },
      });
      // Push the next attempt out so a failed pickup doesn't hot-loop.
      await cases.updateOne(
        { caseId: kase.caseId },
        {
          $set: { nextActionAt: new Date(Date.now() + 24 * 3_600_000) },
          $inc: { attempts: 1 },
        },
      );
      results.push({ caseId: kase.caseId, customer: customer.name, placed: true, call });
    } catch (err) {
      results.push({
        caseId: kase.caseId,
        placed: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return NextResponse.json({ due: due.length, results });
}
