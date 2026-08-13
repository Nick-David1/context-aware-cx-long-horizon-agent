import "dotenv/config";
import { execFileSync } from "node:child_process";
import { collections } from "../src/lib/mongo";

/**
 * Puts the demo back to its opening state and prints proof.
 *
 * Run this between takes. A conversation that changes a payment date also
 * verifies the customer and starts the 90-day clock, so the *second* run of the
 * same demo hits the eligibility block instead of the happy path — which looks
 * like a bug on stage but is the rules working correctly.
 */

execFileSync("npx", ["tsx", "scripts/seed.ts"], { stdio: "inherit" });

const { customers, loans, cases, checkpoints } = await collections();

// A stale in-flight checkpoint would be abandoned on the next turn anyway, but
// clear it so the first turn after a reset starts from a clean slate.
await checkpoints.deleteMany({});

const loan = await loans.findOne({ loanId: "LN-5001" });
const dana = await customers.findOne({ customerId: "CUST-1001" });
const kase = await cases.findOne({ caseId: "CASE-DEMO0001" });

const checks: [string, unknown, unknown][] = [
  ["payment day", loan?.paymentDayOfMonth, 3],
  ["past due", loan?.pastDueAmount, 187.4],
  ["90-day clock", loan?.lastPaymentDateChangeAt ?? null, null],
  ["identity verified", dana?.identityVerified, false],
  ["case status", kase?.status, "open"],
  ["plan steps", kase?.plan.length, 0],
];

console.log("\nopening state:");
let ok = true;
for (const [label, actual, expected] of checks) {
  const pass = String(actual) === String(expected);
  ok &&= pass;
  console.log(`  ${pass ? "✓" : "✗"} ${label.padEnd(18)} ${actual} ${pass ? "" : `(expected ${expected})`}`);
}

console.log(
  ok
    ? "\nReady. CASE-DEMO0001 · Dana · DOB 1988-04-12 · SSN 4417\n"
    : "\nSomething did not reset — check the rows marked ✗ above.\n",
);
process.exit(ok ? 0 : 1);
