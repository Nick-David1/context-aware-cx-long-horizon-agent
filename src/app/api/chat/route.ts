import { NextResponse } from "next/server";
import { runAgentTurn } from "@/lib/agent";

export const runtime = "nodejs";
export const maxDuration = 60;

/** Text channel for the same orchestrator the voice agent uses. Drives the dashboard. */
export async function POST(req: Request) {
  const body = (await req.json()) as {
    caseId: string;
    message: string;
    history?: { role: "customer" | "agent"; text: string }[];
  };

  if (!body.caseId || !body.message) {
    return NextResponse.json({ error: "caseId and message are required" }, { status: 400 });
  }

  const result = await runAgentTurn({
    caseId: body.caseId,
    customerMessage: body.message,
    channel: "chat",
    history: body.history,
  });

  return NextResponse.json({
    reply: result.reply,
    actions: result.actions,
    memoriesUsed: result.memoriesUsed,
  });
}
