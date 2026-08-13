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
  /** OpenAI's opt-in for a trailing usage chunk. ElevenLabs sets it. */
  stream_options?: { include_usage?: boolean };
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

  // Log the shape of what the caller actually sent. ElevenLabs failures surface
  // only as an opaque "LLM Cascade Error" on their side, so the request shape
  // is the only ground truth available for debugging.
  console.log(
    `[llm] ${req.method} stream=${body.stream} msgs=${body.messages?.length} ` +
      `case=${caseId} params=${Object.keys(body).join(",")}`,
  );

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

        // Strictly conventional OpenAI chunk sequence: a role-only chunk first,
        // then content, then a finish chunk. Do NOT use SSE comment lines (":
        // keepalive") to hold the connection — they are valid SSE but many
        // OpenAI-compatible parsers reject a stream that opens with one.
        //
        // The role chunk doubles as the keepalive: it goes out immediately, so
        // the socket carries real data while the tool loop runs, and empty
        // content deltas below extend that without adding to the transcript.
        chunk({ role: "assistant", content: "" }, null);

        // Keep chunks flowing for the WHOLE turn, not just until the first
        // token. A turn speaks, runs tools for several seconds, then speaks
        // again — and that middle gap is silence on the wire. Killing the
        // keepalive at first token left exactly that gap unguarded, which is
        // what "Generating the LLM response took too long" was measuring.
        let lastChunkAt = Date.now();
        const keepalive = setInterval(() => {
          if (Date.now() - lastChunkAt >= 1500) {
            chunk({ content: "" }, null);
            lastChunkAt = Date.now();
          }
        }, 750);

        let streamed = 0;
        const onText = (delta: string) => {
          streamed += delta.length;
          lastChunkAt = Date.now();
          chunk({ content: delta }, null);
        };

        try {
          const result = await runAgentTurn({
            caseId,
            customerMessage,
            channel: "voice",
            history,
            onText,
          });
          clearInterval(keepalive);
          // Only send the assembled reply if nothing streamed — otherwise this
          // would say everything twice.
          const reply = result.reply?.trim() ?? "";
          if (!streamed) {
            chunk({ content: reply || "Sorry, could you say that again?" }, null);
          }

          if (result.endCall) {
            // Hanging up is ElevenLabs' end_call system tool, invoked the same
            // way any OpenAI tool call is: a tool_calls delta, then a
            // finish_reason of "tool_calls". The closing line is spoken from
            // the content chunk above, so `message` is left to the agent's text.
            chunk(
              {
                tool_calls: [
                  {
                    index: 0,
                    id: `call_${Date.now()}`,
                    type: "function",
                    function: {
                      name: "end_call",
                      arguments: JSON.stringify({ reason: result.endCall.reason }),
                    },
                  },
                ],
              },
              null,
            );
            chunk({}, "tool_calls");
            console.log(
              `[llm] case=${caseId} END_CALL ${result.endCall.category}: ${result.endCall.reason}`,
            );
          } else {
            chunk({}, "stop");
          }
          console.log(
            `[llm] case=${caseId} replied ${reply.length} chars (${streamed} streamed)`,
          );
        } catch (err) {
          clearInterval(keepalive);
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

        // ElevenLabs sets stream_options.include_usage, and an OpenAI client
        // that asks for the usage chunk can reject a stream that omits it.
        if (body.stream_options?.include_usage) {
          enqueue(
            `data: ${JSON.stringify({
              id,
              object: "chat.completion.chunk",
              created,
              model: "cx-agent",
              choices: [],
              usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            })}\n\n`,
          );
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
