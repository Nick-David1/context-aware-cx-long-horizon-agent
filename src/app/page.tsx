"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface CaseRow {
  caseId: string;
  customerId: string;
  customerName: string;
  goal: string;
  status: string;
  attempts: number;
  nextActionAt?: string;
  nextActionReason?: string;
  plan: { id: string; description: string; status: string; rationale?: string }[];
  loan: { loanId: string; product: string; balance: number; pastDueAmount: number } | null;
}

interface RecalledMemory {
  memoryId: string;
  kind: "policy" | "episodic" | "lesson";
  text: string;
  score: number;
  rerankScore?: number;
  winRate?: number;
}

interface Pipeline {
  retrieval: "hybrid" | "vector";
  embedding: "auto" | "client";
  rerank: "native" | "client" | "off";
}

interface TurnAction {
  tool: string;
  input: unknown;
  output: unknown;
}

interface ChatTurn {
  role: "customer" | "agent";
  text: string;
}

const KIND_COLOR: Record<string, string> = {
  policy: "#7dd3fc",
  episodic: "#fcd34d",
  lesson: "#5eead4",
};

export default function Dashboard() {
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [memories, setMemories] = useState<RecalledMemory[]>([]);
  const [actions, setActions] = useState<TurnAction[]>([]);
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const loadCases = useCallback(async () => {
    try {
      const res = await fetch("/api/cases");
      const json = (await res.json()) as { cases: CaseRow[] };
      setCases(json.cases ?? []);
      setSelected((s) => s ?? json.cases?.[0]?.caseId ?? null);
    } catch {
      setError("Could not load cases — is MONGODB_URI set and seeded?");
    }
  }, []);

  useEffect(() => {
    void loadCases();
  }, [loadCases]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [turns]);

  const active = cases.find((c) => c.caseId === selected) ?? null;

  async function send() {
    if (!selected || !input.trim() || busy) return;
    const message = input.trim();
    setInput("");
    setTurns((t) => [...t, { role: "customer", text: message }]);
    setBusy(true);
    setError(null);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ caseId: selected, message, history: turns }),
      });
      const json = (await res.json()) as {
        reply?: string;
        actions?: TurnAction[];
        memoriesUsed?: RecalledMemory[];
        pipeline?: Pipeline;
        error?: string;
      };
      if (json.error) throw new Error(json.error);
      setTurns((t) => [...t, { role: "agent", text: json.reply ?? "" }]);
      setMemories(json.memoriesUsed ?? []);
      setActions(json.actions ?? []);
      setPipeline(json.pipeline ?? null);
      void loadCases();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function callNow() {
    if (!selected) return;
    setBusy(true);
    try {
      const res = await fetch("/api/voice/outbound", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ caseId: selected }),
      });
      const json = (await res.json()) as { placed?: boolean; error?: string };
      setError(json.placed ? null : (json.error ?? "Call failed"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid h-screen grid-cols-[300px_1fr_360px] overflow-hidden">
      {/* Case queue */}
      <aside className="overflow-y-auto border-r" style={{ borderColor: "var(--border)" }}>
        <div className="p-4 pb-2">
          <h1 className="text-lg font-semibold tracking-tight">Context-Aware CX</h1>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            Long-horizon servicing agent
          </p>
        </div>
        {cases.map((c) => {
          const due = c.nextActionAt && new Date(c.nextActionAt) <= new Date();
          return (
            <button
              key={c.caseId}
              onClick={() => {
                setSelected(c.caseId);
                setTurns([]);
                setMemories([]);
                setActions([]);
              }}
              className="block w-full border-b px-4 py-3 text-left text-sm transition-colors"
              style={{
                borderColor: "var(--border)",
                background: c.caseId === selected ? "var(--panel)" : "transparent",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium">{c.customerName}</span>
                <span
                  className="rounded px-1.5 py-0.5 text-[10px] uppercase"
                  style={{
                    background: due ? "#7f1d1d" : "var(--border)",
                    color: due ? "#fecaca" : "var(--muted)",
                  }}
                >
                  {due ? "due" : c.status}
                </span>
              </div>
              <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                {c.goal.replace(/_/g, " ")} · {c.caseId}
              </div>
            </button>
          );
        })}
      </aside>

      {/* Conversation */}
      <section className="flex min-w-0 flex-col">
        <header
          className="flex items-center justify-between border-b px-5 py-3"
          style={{ borderColor: "var(--border)" }}
        >
          <div>
            <div className="text-sm font-medium">
              {active ? `${active.customerName} — ${active.goal.replace(/_/g, " ")}` : "No case"}
            </div>
            {active?.loan && (
              <div className="text-xs" style={{ color: "var(--muted)" }}>
                {active.loan.product} {active.loan.loanId} · balance $
                {active.loan.balance.toLocaleString()}
                {active.loan.pastDueAmount > 0 &&
                  ` · past due $${active.loan.pastDueAmount.toFixed(2)}`}
              </div>
            )}
          </div>
          <button
            onClick={callNow}
            disabled={!selected || busy}
            className="rounded px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: "var(--accent)", color: "#04211d" }}
          >
            Call customer
          </button>
        </header>

        {active && active.plan.length > 0 && (
          <div className="border-b px-5 py-3" style={{ borderColor: "var(--border)" }}>
            <div className="mb-1.5 text-[10px] uppercase tracking-wide" style={{ color: "var(--muted)" }}>
              Agent plan {active.nextActionReason && `· next: ${active.nextActionReason}`}
            </div>
            {active.plan.map((s) => (
              <div key={s.id} className="text-xs leading-relaxed">
                <span style={{ color: s.status === "done" ? "var(--accent)" : "var(--muted)" }}>
                  {s.status === "done" ? "✓" : "○"}
                </span>{" "}
                {s.description}
              </div>
            ))}
          </div>
        )}

        <div ref={scroller} className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
          {turns.length === 0 && (
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Speak as the customer to test the agent. The same orchestrator answers the phone.
            </p>
          )}
          {turns.map((t, i) => (
            <div key={i} className={t.role === "customer" ? "text-right" : ""}>
              <div
                className="inline-block max-w-[75%] rounded-lg px-3 py-2 text-sm leading-relaxed"
                style={{
                  background: t.role === "customer" ? "#1e293b" : "var(--panel)",
                  border: "1px solid var(--border)",
                }}
              >
                {t.text}
              </div>
            </div>
          ))}
          {busy && (
            <div className="text-xs" style={{ color: "var(--muted)" }}>
              thinking…
            </div>
          )}
          {error && (
            <div className="rounded border px-3 py-2 text-xs" style={{ borderColor: "#7f1d1d", color: "#fca5a5" }}>
              {error}
            </div>
          )}
        </div>

        <div className="border-t p-3" style={{ borderColor: "var(--border)" }}>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
            className="flex gap-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Say something as the customer…"
              className="flex-1 rounded px-3 py-2 text-sm outline-none"
              style={{ background: "var(--panel)", border: "1px solid var(--border)" }}
            />
            <button
              type="submit"
              disabled={busy}
              className="rounded px-4 py-2 text-sm font-medium disabled:opacity-40"
              style={{ background: "var(--accent)", color: "#04211d" }}
            >
              Send
            </button>
          </form>
        </div>
      </section>

      {/* Context engine inspector */}
      <aside className="overflow-y-auto border-l" style={{ borderColor: "var(--border)" }}>
        <div className="px-4 pt-3 text-[10px] uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          Context retrieved this turn
        </div>
        {pipeline && (
          <div className="flex flex-wrap gap-1 px-4 pb-3 pt-1.5">
            {[
              pipeline.retrieval === "hybrid" ? "$rankFusion" : "$vectorSearch",
              pipeline.embedding === "auto" ? "autoEmbed" : "client embed",
              pipeline.rerank === "native"
                ? "$rerank"
                : pipeline.rerank === "client"
                  ? "rerank (api)"
                  : "no rerank",
            ].map((stage) => (
              <span
                key={stage}
                className="rounded px-1.5 py-0.5 font-mono text-[10px]"
                style={{ background: "var(--panel)", border: "1px solid var(--border)" }}
              >
                {stage}
              </span>
            ))}
          </div>
        )}
        {memories.length === 0 && (
          <p className="px-4 text-xs" style={{ color: "var(--muted)" }}>
            Vector search results appear here after each turn.
          </p>
        )}
        {memories.map((m) => (
          <div key={m.memoryId} className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
            <div className="mb-1 flex items-center gap-2 text-[10px] uppercase">
              <span style={{ color: KIND_COLOR[m.kind] }}>{m.kind}</span>
              <span
                style={{ color: "var(--muted)" }}
                title={
                  pipeline?.retrieval === "hybrid"
                    ? "reciprocal rank fusion score"
                    : "vector similarity"
                }
              >
                {pipeline?.retrieval === "hybrid" ? "rrf" : "vec"} {m.score.toFixed(3)}
              </span>
              {m.rerankScore !== undefined && (
                <span style={{ color: "#c4b5fd" }} title="cross-encoder rerank score">
                  rr {m.rerankScore.toFixed(3)}
                </span>
              )}
              {m.winRate !== undefined && (
                <span style={{ color: m.winRate >= 0.6 ? "var(--accent)" : "#fca5a5" }}>
                  {(m.winRate * 100).toFixed(0)}% win
                </span>
              )}
            </div>
            <div className="text-xs leading-relaxed">{m.text}</div>
          </div>
        ))}

        {actions.length > 0 && (
          <>
            <div
              className="px-4 py-3 text-[10px] uppercase tracking-wide"
              style={{ color: "var(--muted)" }}
            >
              Actions taken
            </div>
            {actions.map((a, i) => (
              <div key={i} className="border-b px-4 py-2" style={{ borderColor: "var(--border)" }}>
                <div className="text-xs" style={{ color: "var(--accent)" }}>
                  {a.tool}
                </div>
                <pre className="mt-1 overflow-x-auto text-[10px]" style={{ color: "var(--muted)" }}>
                  {JSON.stringify(a.output, null, 1).slice(0, 400)}
                </pre>
              </div>
            ))}
          </>
        )}
      </aside>
    </main>
  );
}
