import type { ObjectId } from "mongodb";

/** A person the agent serves. Long-lived, spans many cases and conversations. */
export interface Customer {
  _id?: ObjectId;
  customerId: string;
  name: string;
  phone: string;
  email?: string;
  /** Identity verification is a precondition for every write action. */
  identityVerified: boolean;
  identityVerifiedAt?: Date;
  /** Free-form facts the agent learned about this person over time. */
  traits: string[];
  createdAt: Date;
}

export type LoanStatus = "current" | "delinquent" | "default" | "paid_off";

export interface Loan {
  _id?: ObjectId;
  loanId: string;
  customerId: string;
  product: string;
  principal: number;
  balance: number;
  aprBps: number;
  termMonths: number;
  monthlyPayment: number;
  /** Day of month the payment is drafted, 1-28. */
  paymentDayOfMonth: number;
  status: LoanStatus;
  latePayments12mo: number;
  pastDueAmount: number;
  lastPaymentDateChangeAt?: Date;
  openedAt: Date;
}

/** The long-horizon goal. Survives across days, calls, and channels. */
export type CaseGoal =
  | "change_payment_date"
  | "refinance"
  | "collect_past_due"
  | "hardship_plan"
  | "payoff_quote";

export type CaseStatus = "open" | "waiting_on_customer" | "won" | "lost" | "escalated";

export interface PlanStep {
  id: string;
  description: string;
  status: "pending" | "done" | "blocked" | "skipped";
  /** Why the agent chose this step — shown in the dashboard, fed back into learning. */
  rationale?: string;
  completedAt?: Date;
}

export interface CaseRecord {
  _id?: ObjectId;
  caseId: string;
  customerId: string;
  loanId?: string;
  goal: CaseGoal;
  status: CaseStatus;
  /** The agent's current plan. Rewritten between engagements, not hard-coded. */
  plan: PlanStep[];
  /** When the agent should proactively reach out again. Drives the outbound tick. */
  nextActionAt?: Date;
  nextActionReason?: string;
  attempts: number;
  outcome?: {
    result: "won" | "lost";
    reason: string;
    closedAt: Date;
  };
  createdAt: Date;
  updatedAt: Date;
}

export type Channel = "voice" | "chat" | "sms";

export interface Interaction {
  _id?: ObjectId;
  interactionId: string;
  caseId: string;
  customerId: string;
  channel: Channel;
  direction: "inbound" | "outbound";
  turns: { role: "customer" | "agent" | "system"; text: string; at: Date }[];
  /** Tool calls the agent made during this interaction, for the audit trail. */
  actions: { tool: string; input: unknown; output: unknown; at: Date }[];
  startedAt: Date;
  endedAt?: Date;
}

export type MemoryKind =
  /** Company policy / business logic. Authored, versioned, supersedable. */
  | "policy"
  /** Something that actually happened with this customer. */
  | "episodic"
  /** A generalization the agent derived from outcomes. This is the compounding moat. */
  | "lesson";

export interface Memory {
  _id?: ObjectId;
  memoryId: string;
  kind: MemoryKind;
  /** null = applies to every customer. */
  customerId: string | null;
  /**
   * `customerId` collapsed to a non-null token: the customer id, or "global".
   *
   * Exists because the two indexes disagree about null. $vectorSearch takes an
   * MQL filter where `customerId: null` matches fine; Lucene has no clean way
   * to equal-match a null token, so BM25 filtering needs a real value. One
   * derived field lets both indexes use the identical filter.
   */
  scope: string;
  text: string;
  /**
   * Only present in client embedding mode. In auto mode the document is written
   * without it and MongoDB vectorizes `text` on write.
   */
  embedding?: number[];
  /** Tags used as Atlas Vector Search pre-filters. */
  tags: string[];
  /** Lessons track how they performed so retrieval can prefer what works. */
  stats?: { timesApplied: number; wins: number; losses: number };
  /** False once superseded. Used as a vector-search pre-filter so stale context is never retrieved. */
  active: boolean;
  /** Set when a newer memory replaces this one — keeps context current, not just additive. */
  supersededBy?: string;
  sourceCaseIds?: string[];
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Durable working state for an in-flight turn.
 *
 * The agent's tool loop can run many steps against live systems. If the process
 * dies partway through, the checkpoint is what lets the next turn resume from
 * the last completed tool call instead of replaying side effects or starting the
 * case over. This is the "no cold start" guarantee at the turn level.
 */
export interface Checkpoint {
  _id?: ObjectId;
  caseId: string;
  /** Anthropic message array as of the last completed step. */
  messages: unknown[];
  /** Tool calls already executed this turn — replay guard. */
  completedTools: { tool: string; input: unknown; output: unknown; at: Date }[];
  step: number;
  status: "in_flight" | "complete" | "abandoned";
  updatedAt: Date;
}

/** Terminal signal for a case. Reflection reads these to write lessons. */
export interface Outcome {
  _id?: ObjectId;
  outcomeId: string;
  caseId: string;
  customerId: string;
  goal: CaseGoal;
  result: "won" | "lost";
  reason: string;
  /** Memory ids that were in context when the agent acted — credit assignment. */
  memoriesUsed: string[];
  createdAt: Date;
}
