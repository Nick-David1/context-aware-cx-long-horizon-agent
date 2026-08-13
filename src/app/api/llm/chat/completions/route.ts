import { NextResponse } from "next/server";
import { runAgentTurn } from "@/lib/agent";

export const runtime = "nodejs";
export const maxDuration = 60;

/**
 * OpenAI-compatible chat-completions endpoint.
 *
 * ElevenLabs Conversational AI points here as its "custom LLM": it handles
 * speech-to-text, turn-taking, and text-to-speech, and calls this for every
 * customer utterance. This service supplies the reasoning, memory, and account
 * actions — so the voice agent has the full case context automatically, with
 * no per-tool webhook plumbing.
 */

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatRequest {
  messages: ChatMessage[];
  stream?: boolean;
  /** ElevenLabs passes extra body params through; caseId can arrive here. */
  caseId?: string;
}

/** The case id rides in on the system prompt as `case:CASE-XXXXXXX`. */
function extractCaseId(body: ChatRequest): string | null {
  if (body.caseId) return body.caseId;
  for (const m of body.messages) {
    const match = m.content?.match(/case:\s*(CASE-[A-Z0-9]+)/i);
    if (match) return match[1].toUpperCase();
  }
  return null;
}

export async function POST(req: Request) {
  // This endpoint is reachable from the public internet through the tunnel, and
  // it can spend model tokens and mutate loan records. ElevenLabs sends the
  // shared secret as a custom request header (custom_llm.request_headers).
  const expected = process.env.TOOL_WEBHOOK_SECRET;
  if (expected && req.headers.get("x-cx-secret") !== expected) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const body = (await req.json()) as ChatRequest;
  const caseId = extractCaseId(body);

  if (!caseId) {
    return NextResponse.json(
      { error: "No caseId. Put `case:{{caseId}}` in the ElevenLabs agent prompt." },
      { status: 400 },
    );
  }

  const conversation = body.messages.filter((m) => m.role !== "system");
  const last = conversation.at(-1);
  const customerMessage = last?.role === "user" ? last.content : "";
  const history = conversation
    .slice(0, last?.role === "user" ? -1 : undefined)
    .map((m) => ({ role: m.role === "user" ? ("customer" as const) : ("agent" as const), text: m.content }));

  const result = await runAgentTurn({
    caseId,
    customerMessage,
    channel: "voice",
    history,
  });

  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  if (!body.stream) {
    return NextResponse.json({
      id,
      object: "chat.completion",
      created,
      model: "cx-agent",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: result.reply },
          finish_reason: "stop",
        },
      ],
    });
  }

  // ElevenLabs expects SSE. The orchestrator's tool loop has already finished by
  // this point, so we emit the finished reply as a single chunk plus a terminator
  // rather than faking token-by-token streaming.
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      const chunk = (delta: Record<string, unknown>, finish: string | null) =>
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              id,
              object: "chat.completion.chunk",
              created,
              model: "cx-agent",
              choices: [{ index: 0, delta, finish_reason: finish }],
            })}\n\n`,
          ),
        );

      chunk({ role: "assistant", content: result.reply }, null);
      chunk({}, "stop");
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    },
  });
}
