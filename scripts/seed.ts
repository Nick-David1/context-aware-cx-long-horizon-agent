import "dotenv/config";
import { collections } from "../src/lib/mongo";
import { remember } from "../src/lib/memory";
import { usingRealEmbeddings } from "../src/lib/embeddings";
import { config } from "../src/lib/config";
import type { Customer, CaseRecord, Loan } from "../src/lib/types";

/**
 * Seeds a demo book of business: three customers, their loans, open cases, the
 * policy corpus, and a handful of pre-existing lessons so the "it already knows
 * things" story is visible on the very first run.
 */

const now = new Date();
const daysAgo = (d: number) => new Date(now.getTime() - d * 86_400_000);
const hoursFromNow = (h: number) => new Date(now.getTime() + h * 3_600_000);

const customers: (Customer & { dobYYYYMMDD: string; ssnLast4: string })[] = [
  {
    customerId: "CUST-1001",
    name: "Dana Whitfield",
    phone: "+15551230001",
    email: "dana@example.com",
    identityVerified: false,
    traits: ["Paid on time for 3 years", "Prefers evening calls"],
    dobYYYYMMDD: "1988-04-12",
    ssnLast4: "4417",
    createdAt: daysAgo(1100),
  },
  {
    customerId: "CUST-1002",
    name: "Marcus Oyelaran",
    phone: "+15551230002",
    email: "marcus@example.com",
    identityVerified: false,
    traits: ["Recently changed jobs", "Two missed payments this year"],
    dobYYYYMMDD: "1979-11-03",
    ssnLast4: "8820",
    createdAt: daysAgo(600),
  },
  {
    customerId: "CUST-1003",
    name: "Priya Raman",
    phone: "+15551230003",
    email: "priya@example.com",
    identityVerified: false,
    traits: ["Shopping competitor refinance offers"],
    dobYYYYMMDD: "1992-07-22",
    ssnLast4: "0134",
    createdAt: daysAgo(320),
  },
];

const loans: Loan[] = [
  {
    loanId: "LN-5001",
    customerId: "CUST-1001",
    product: "Auto Loan",
    principal: 32000,
    balance: 18450.22,
    aprBps: 749,
    termMonths: 60,
    monthlyPayment: 612.4,
    paymentDayOfMonth: 3,
    status: "current",
    latePayments12mo: 0,
    pastDueAmount: 0,
    openedAt: daysAgo(900),
  },
  {
    loanId: "LN-5002",
    customerId: "CUST-1002",
    product: "Personal Loan",
    principal: 15000,
    balance: 11230.55,
    aprBps: 1425,
    termMonths: 48,
    monthlyPayment: 412.1,
    paymentDayOfMonth: 15,
    status: "delinquent",
    latePayments12mo: 2,
    pastDueAmount: 412.1,
    openedAt: daysAgo(480),
  },
  {
    loanId: "LN-5003",
    customerId: "CUST-1003",
    product: "Auto Loan",
    principal: 41000,
    balance: 29875.0,
    aprBps: 1090,
    termMonths: 72,
    monthlyPayment: 588.25,
    paymentDayOfMonth: 20,
    status: "current",
    latePayments12mo: 1,
    pastDueAmount: 0,
    openedAt: daysAgo(300),
  },
];

const cases: CaseRecord[] = [
  {
    caseId: "CASE-DEMO0001",
    customerId: "CUST-1001",
    loanId: "LN-5001",
    goal: "change_payment_date",
    status: "open",
    plan: [],
    attempts: 0,
    createdAt: daysAgo(1),
    updatedAt: daysAgo(1),
  },
  {
    caseId: "CASE-DEMO0002",
    customerId: "CUST-1002",
    loanId: "LN-5002",
    goal: "collect_past_due",
    status: "waiting_on_customer",
    plan: [
      {
        id: "step-1",
        description: "Reach Marcus and understand why the payment was missed",
        status: "done",
        rationale: "Cause determines whether this is a hardship case or a timing problem",
        completedAt: daysAgo(2),
      },
      {
        id: "step-2",
        description: "Offer to split the past-due amount across two payments",
        status: "pending",
        rationale: "He said cash is tight until his new job's first full paycheck",
      },
    ],
    // Already due, so `npm run tick` has something to do immediately.
    nextActionAt: hoursFromNow(-1),
    nextActionReason: "He asked to be called back after payday on the 15th",
    attempts: 1,
    createdAt: daysAgo(5),
    updatedAt: daysAgo(2),
  },
  {
    caseId: "CASE-DEMO0003",
    customerId: "CUST-1003",
    loanId: "LN-5003",
    goal: "refinance",
    status: "open",
    plan: [],
    attempts: 0,
    createdAt: daysAgo(3),
    updatedAt: daysAgo(3),
  },
];

/** Authored business logic — the "local context" a generic model cannot know. */
const policies: { text: string; tags: string[] }[] = [
  {
    text: "Identity must be verified with date of birth and the last four of the SSN before any change to an account. Payoff quotes are the only exception.",
    tags: ["identity", "compliance"],
  },
  {
    text: "Payment date changes are allowed once per 90 days, to any day from the 1st through the 28th. The loan must be current with no past-due balance.",
    tags: ["change_payment_date"],
  },
  {
    text: "Refinance requires the loan to be current, a remaining balance of at least $5,000, and no more than one late payment in the past twelve months.",
    tags: ["refinance"],
  },
  {
    text: "Hardship plans reduce the payment for three to six months and are available to any customer whose loan is not paid off. Enrolling pauses late fees but interest continues to accrue.",
    tags: ["hardship_plan"],
  },
  {
    text: "Never quote an interest rate the customer has not been formally approved for. Approved rates come from a refinance application, not from a conversation.",
    tags: ["refinance", "compliance"],
  },
  {
    text: "A customer who says the words 'financial hardship' must be offered the hardship plan before any collection pressure is applied.",
    tags: ["collect_past_due", "hardship_plan", "compliance"],
  },
  {
    text: "Payoff quotes are calculated with ten days of daily interest added to the current balance and are valid for ten days.",
    tags: ["payoff_quote"],
  },
];

/** Lessons "learned" from earlier cases, with track records already attached. */
const lessons: { text: string; tags: string[]; wins: number; losses: number }[] = [
  {
    text: "Customers asking to move their payment date are usually a few days short at month end. Confirm the loan is current first — if there is a past-due balance the change is blocked, and discovering that after you have offered it costs the call.",
    tags: ["change_payment_date"],
    wins: 7,
    losses: 1,
  },
  {
    text: "On refinance calls, ask what rate they have been quoted elsewhere before pitching. Customers who name a competitor rate convert at a much higher rate once you start the application on the call rather than promising a callback.",
    tags: ["refinance"],
    wins: 5,
    losses: 2,
  },
  {
    text: "For past-due collection, opening with the reason for the miss rather than the amount owed keeps the customer on the line. Leading with the dollar figure ends calls early.",
    tags: ["collect_past_due"],
    wins: 9,
    losses: 2,
  },
  {
    text: "Offering a hardship plan to anyone who mentions a job change resolves the case faster than a payment plan, even when they did not use the word hardship.",
    tags: ["collect_past_due", "hardship_plan"],
    wins: 2,
    losses: 4,
  },
];

async function main() {
  const {
    customers: cColl,
    loans: lColl,
    cases: caseColl,
    memories,
    interactions,
    outcomes,
    checkpoints,
  } = await collections();

  console.log(
    usingRealEmbeddings()
      ? `Embeddings: ${config.embeddingMode} mode${config.embeddingMode === "auto" ? " — Atlas vectorizes on write" : ` via ${config.embedModel}`}.`
      : "VOYAGE_API_KEY unset in client mode — using the local fallback embedder (much worse retrieval).",
  );

  await Promise.all([
    cColl.deleteMany({}),
    lColl.deleteMany({}),
    caseColl.deleteMany({}),
    memories.deleteMany({}),
    interactions.deleteMany({}),
    outcomes.deleteMany({}),
    checkpoints.deleteMany({}),
  ]);

  await cColl.insertMany(customers);
  await lColl.insertMany(loans);
  await caseColl.insertMany(cases);
  console.log(`seeded ${customers.length} customers, ${loans.length} loans, ${cases.length} cases`);

  for (const p of policies) {
    await remember({ kind: "policy", customerId: null, text: p.text, tags: p.tags });
  }
  console.log(`seeded ${policies.length} policies`);

  for (const l of lessons) {
    const m = await remember({ kind: "lesson", customerId: null, text: l.text, tags: l.tags });
    await memories.updateOne(
      { memoryId: m.memoryId },
      { $set: { stats: { timesApplied: l.wins + l.losses, wins: l.wins, losses: l.losses } } },
    );
  }
  console.log(`seeded ${lessons.length} lessons with track records`);

  // Some prior history, so the agent has something customer-specific to recall.
  await remember({
    kind: "episodic",
    customerId: "CUST-1002",
    text: "Marcus said he started a new job in March and his first full paycheck lands on the 15th. He asked not to be called before then.",
    tags: ["collect_past_due"],
    sourceCaseIds: ["CASE-DEMO0002"],
  });
  await remember({
    kind: "episodic",
    customerId: "CUST-1003",
    text: "Priya mentioned a credit union quoted her 6.9 percent on an auto refinance and she is comparing offers this week.",
    tags: ["refinance"],
    sourceCaseIds: ["CASE-DEMO0003"],
  });

  console.log("\nDone. Try: CASE-DEMO0001 (payment date), CASE-DEMO0002 (collections), CASE-DEMO0003 (refinance)");
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
