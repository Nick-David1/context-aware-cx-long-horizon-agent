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

  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  if (body.stream) {
    // Open the SSE response IMMEDIATELY and heartbeat while the agent works.
    //
    // A turn that verifies identity, checks eligibility, and executes an action
    // is four-plus model round trips — 15s or more. Awaiting the whole loop
    // before returning leaves the socket silent that long, and ElevenLabs drops
    // it with "custom_llm generation failed". SSE comment lines (": ...") are
    // ignored by every SSE parser, so they hold the connection open without
    // putting a single token into the transcript.
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        const enqueue = (s: string) => {
          try {
            controller.enqueue(encoder.encode(s));
          } catch {
            // Client hung up; the loop below still finishes and persists state.
          }
        };

        const heartbeat = setInterval(() => enqueue(": keepalive\n\n"), 2000);
        enqueue(": open\n\n");

        const chunk = (delta: Record<string, unknown>, finish: string | null) =>
          enqueue(
            `data: ${JSON.stringify({
              id,
              object: "chat.completion.chunk",
              created,
              model: "cx-agent",
              choices: [{ index: 0, delta, finish_reason: finish }],
            })}\n\n`,
          );

        try {
          const result = await runAgentTurn({
            caseId,
            customerMessage,
            channel: "voice",
            history,
          });
          clearInterval(heartbeat);
          chunk({ role: "assistant", content: result.reply }, null);
          chunk({}, "stop");
        } catch (err) {
          clearInterval(heartbeat);
          console.error("agent turn failed:", err);
          // Say something rather than nothing — a silent stream is what makes
          // ElevenLabs fail the whole conversation.
          chunk(
            {
              role: "assistant",
              content:
                "Sorry, I hit a problem pulling up your account. Can you say that again?",
            },
            null,
          );
          chunk({}, "stop");
        }

        enqueue("data: [DONE]\n\n");
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
        // Stops any proxy in the path from buffering the heartbeats.
        "x-accel-buffering": "no",
      },
    });
  }

  {
    const result = await runAgentTurn({
      caseId,
      customerMessage,
      channel: "voice",
      history,
    });
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
}
