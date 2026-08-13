import { randomUUID } from "node:crypto";
import { collections } from "./mongo";
import { remember, scoreLessons } from "./memory";
import type { CaseGoal, Customer, CaseRecord, Loan, PlanStep } from "./types";

/**
 * Write actions the agent can take, plus the deterministic eligibility rules that
 * gate them.
 *
 * Validation lives in code, not in the prompt. The model proposes an action; this
 * module decides whether it is allowed. That split is what makes write operations
 * safe to hand to an LLM — a hallucinated proposal fails validation instead of
 * moving money.
 */

export type ActionName =
  | "change_payment_date"
  | "start_refinance_application"
  | "enroll_hardship_plan"
  | "issue_payoff_quote"
  | "take_payment";

export interface ActionInput {
  action: ActionName;
  loanId: string;
  /** Action-specific parameters. Validated per-action below. */
  params: Record<string, unknown>;
}

export interface ValidationResult {
  allowed: boolean;
  /** Human-readable reasons, surfaced to the customer and logged on the case. */
  reasons: string[];
  /** What the customer would need to do to become eligible. */
  remediation?: string;
}

export async function loadCustomerAndLoan(
  customerId: string,
  loanId?: string,
): Promise<{ customer: Customer | null; loans: Loan[]; loan: Loan | null }> {
  const { customers, loans: loanColl } = await collections();
  const customer = await customers.findOne({ customerId });
  const loans = await loanColl.find({ customerId }).toArray();
  const loan = loanId ? (loans.find((l) => l.loanId === loanId) ?? null) : (loans[0] ?? null);
  return { customer, loans, loan };
}

export async function validateAction(
  customerId: string,
  input: ActionInput,
): Promise<ValidationResult> {
  const { customer, loan } = await loadCustomerAndLoan(customerId, input.loanId);
  const reasons: string[] = [];

  if (!customer) return { allowed: false, reasons: ["Customer not found."] };
  if (!loan) return { allowed: false, reasons: [`Loan ${input.loanId} not found.`] };

  // Identity verification gates every write. Read-only quotes are exempt.
  const readOnly = input.action === "issue_payoff_quote";
  if (!readOnly && !customer.identityVerified) {
    return {
      allowed: false,
      reasons: ["Identity is not verified."],
      remediation:
        "Verify identity first: confirm date of birth and the last four digits of the SSN.",
    };
  }

  switch (input.action) {
    case "change_payment_date": {
      const day = Number(input.params.newPaymentDayOfMonth);
      if (!Number.isInteger(day) || day < 1 || day > 28) {
        reasons.push("New payment day must be a whole number between 1 and 28.");
      }
      if (loan.status === "default") {
        reasons.push("Payment date changes are not allowed on loans in default.");
      }
      if (loan.pastDueAmount > 0) {
        return {
          allowed: false,
          reasons: [`Loan is past due by $${loan.pastDueAmount.toFixed(2)}.`],
          remediation: "Bring the loan current, then the payment date can be changed.",
        };
      }
      if (loan.lastPaymentDateChangeAt) {
        const daysSince =
          (Date.now() - loan.lastPaymentDateChangeAt.getTime()) / 86_400_000;
        if (daysSince < 90) {
          reasons.push(
            `Payment date was changed ${Math.floor(daysSince)} days ago; one change is allowed per 90 days.`,
          );
        }
      }
      break;
    }
    case "start_refinance_application": {
      if (loan.status !== "current") {
        reasons.push("Refinance requires the loan to be current.");
      }
      if (loan.latePayments12mo > 1) {
        reasons.push(
          `Refinance requires at most 1 late payment in 12 months; this loan has ${loan.latePayments12mo}.`,
        );
      }
      if (loan.balance < 5000) {
        reasons.push("Refinance requires a remaining balance of at least $5,000.");
      }
      break;
    }
    case "enroll_hardship_plan": {
      if (loan.status === "paid_off") reasons.push("Loan is already paid off.");
      break;
    }
    case "take_payment": {
      const amount = Number(input.params.amount);
      if (!Number.isFinite(amount) || amount <= 0) {
        reasons.push("Payment amount must be greater than zero.");
      }
      break;
    }
    case "issue_payoff_quote":
      break;
  }

  return { allowed: reasons.length === 0, reasons };
}

export interface ExecutionResult {
  executed: boolean;
  summary: string;
  details?: Record<string, unknown>;
}

export async function executeAction(
  customerId: string,
  caseId: string,
  input: ActionInput,
): Promise<ExecutionResult> {
  const validation = await validateAction(customerId, input);
  if (!validation.allowed) {
    return {
      executed: false,
      summary: `Blocked: ${validation.reasons.join(" ")}`,
      details: { remediation: validation.remediation },
    };
  }

  const { loans } = await collections();
  const { loan } = await loadCustomerAndLoan(customerId, input.loanId);
  if (!loan) return { executed: false, summary: "Loan not found." };

  let summary: string;
  let details: Record<string, unknown> = {};

  switch (input.action) {
    case "change_payment_date": {
      const day = Number(input.params.newPaymentDayOfMonth);
      await loans.updateOne(
        { loanId: loan.loanId },
        { $set: { paymentDayOfMonth: day, lastPaymentDateChangeAt: new Date() } },
      );
      summary = `Payment date moved from the ${loan.paymentDayOfMonth}th to the ${day}th of each month.`;
      details = { previousDay: loan.paymentDayOfMonth, newDay: day };
      break;
    }
    case "start_refinance_application": {
      const applicationId = `APP-${randomUUID().slice(0, 8).toUpperCase()}`;
      summary = `Refinance application ${applicationId} opened for loan ${loan.loanId}.`;
      details = { applicationId };
      break;
    }
    case "enroll_hardship_plan": {
      const months = Number(input.params.months ?? 3);
      summary = `Enrolled in a ${months}-month hardship plan with reduced payments.`;
      details = { months };
      break;
    }
    case "issue_payoff_quote": {
      // Simple daily-interest payoff good for 10 days.
      const dailyRate = loan.aprBps / 10_000 / 365;
      const payoff = loan.balance * (1 + dailyRate * 10);
      summary = `Payoff quote: $${payoff.toFixed(2)}, good for 10 days.`;
      details = { payoff: Number(payoff.toFixed(2)), goodForDays: 10 };
      break;
    }
    case "take_payment": {
      const amount = Number(input.params.amount);
      const newBalance = Math.max(0, loan.balance - amount);
      const newPastDue = Math.max(0, loan.pastDueAmount - amount);
      await loans.updateOne(
        { loanId: loan.loanId },
        {
          $set: {
            balance: newBalance,
            pastDueAmount: newPastDue,
            status: newBalance === 0 ? "paid_off" : newPastDue === 0 ? "current" : loan.status,
          },
        },
      );
      summary = `Payment of $${amount.toFixed(2)} applied. New balance $${newBalance.toFixed(2)}.`;
      details = { amount, newBalance, newPastDue };
      break;
    }
  }

  // Every write becomes an episodic memory so future conversations know it happened.
  await remember({
    kind: "episodic",
    customerId,
    text: `${new Date().toISOString().slice(0, 10)} — ${summary}`,
    tags: [input.action, "action"],
    sourceCaseIds: [caseId],
  });

  return { executed: true, summary, details };
}

export async function verifyIdentity(
  customerId: string,
  provided: { dobYYYYMMDD?: string; ssnLast4?: string },
): Promise<{ verified: boolean; reason: string }> {
  const { customers } = await collections();
  const customer = await customers.findOne({ customerId });
  if (!customer) return { verified: false, reason: "Customer not found." };

  // Demo-grade check. In production this calls the bank's KYC service; the point
  // here is that verification is a gate the agent cannot talk its way past.
  const record = customer as Customer & { dobYYYYMMDD?: string; ssnLast4?: string };
  const dobOk = !record.dobYYYYMMDD || record.dobYYYYMMDD === provided.dobYYYYMMDD;
  const ssnOk = !record.ssnLast4 || record.ssnLast4 === provided.ssnLast4;

  if (!dobOk || !ssnOk) {
    return { verified: false, reason: "Provided details do not match our records." };
  }

  await customers.updateOne(
    { customerId },
    { $set: { identityVerified: true, identityVerifiedAt: new Date() } },
  );
  return { verified: true, reason: "Identity verified." };
}

export async function updatePlan(
  caseId: string,
  plan: PlanStep[],
  nextAction?: { at: Date; reason: string },
): Promise<void> {
  const { cases } = await collections();
  await cases.updateOne(
    { caseId },
    {
      $set: {
        plan,
        updatedAt: new Date(),
        ...(nextAction
          ? { nextActionAt: nextAction.at, nextActionReason: nextAction.reason }
          : {}),
      },
    },
  );
}

export async function closeCase(
  caseId: string,
  result: "won" | "lost",
  reason: string,
  memoriesUsed: string[],
): Promise<void> {
  const { cases, outcomes } = await collections();
  const kase = await cases.findOne({ caseId });
  if (!kase) throw new Error(`Case ${caseId} not found`);

  const now = new Date();
  await cases.updateOne(
    { caseId },
    {
      $set: {
        status: result,
        outcome: { result, reason, closedAt: now },
        nextActionAt: undefined,
        updatedAt: now,
      },
    },
  );

  await outcomes.insertOne({
    outcomeId: randomUUID(),
    caseId,
    customerId: kase.customerId,
    goal: kase.goal,
    result,
    reason,
    memoriesUsed,
    createdAt: now,
  });

  // Credit assignment happens immediately; reflection (which writes new lessons)
  // runs separately so a slow model call never blocks closing a case.
  await scoreLessons(memoriesUsed, result);
}

export async function openCase(
  customerId: string,
  goal: CaseGoal,
  loanId?: string,
): Promise<CaseRecord> {
  const { cases } = await collections();
  const now = new Date();
  const kase: CaseRecord = {
    caseId: `CASE-${randomUUID().slice(0, 8).toUpperCase()}`,
    customerId,
    loanId,
    goal,
    status: "open",
    plan: [],
    attempts: 0,
    createdAt: now,
    updatedAt: now,
  };
  await cases.insertOne(kase);
  return kase;
}
