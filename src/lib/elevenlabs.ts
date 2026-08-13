const BASE = "https://api.elevenlabs.io";

function apiKey(): string {
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) throw new Error("ELEVENLABS_API_KEY is not set");
  return key;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "xi-api-key": apiKey(),
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`ElevenLabs ${path} failed (${res.status}): ${await res.text()}`);
  }
  return (await res.json()) as T;
}

/**
 * Places an outbound call through the ElevenLabs Conversational AI agent.
 * This is what turns a scheduled follow-up into an actual phone call.
 */
export async function startOutboundCall(args: {
  toNumber: string;
  /** Injected into the agent's prompt as {{caseId}} etc. */
  variables: Record<string, string>;
}): Promise<{ callSid?: string; conversationId?: string }> {
  const agentId = process.env.ELEVENLABS_AGENT_ID;
  const phoneNumberId = process.env.ELEVENLABS_PHONE_NUMBER_ID;
  if (!agentId || !phoneNumberId) {
    throw new Error("ELEVENLABS_AGENT_ID and ELEVENLABS_PHONE_NUMBER_ID must be set");
  }

  return call("/v1/convai/twilio/outbound-call", {
    method: "POST",
    body: JSON.stringify({
      agent_id: agentId,
      agent_phone_number_id: phoneNumberId,
      to_number: args.toNumber,
      conversation_initiation_client_data: {
        dynamic_variables: args.variables,
      },
    }),
  });
}

/**
 * Points an existing ElevenLabs agent at this deployment as its custom LLM.
 *
 * Run this whenever the tunnel URL changes — a cloudflared quick tunnel gets a
 * new hostname on every restart, and the agent stores the URL, so a stale one
 * means the agent connects and then says nothing.
 *
 * `caseId` defaults to the `{{caseId}}` dynamic variable, which outbound calls
 * populate per call. Pass a literal case id to pin the agent to one case for
 * widget testing, where there is no outbound call to inject it.
 */
export async function syncAgentToCustomLLM(
  baseUrl: string,
  caseId = "{{caseId}}",
): Promise<unknown> {
  const agentId = process.env.ELEVENLABS_AGENT_ID;
  if (!agentId) throw new Error("ELEVENLABS_AGENT_ID is not set");

  const secret = process.env.TOOL_WEBHOOK_SECRET;
  if (!secret) throw new Error("TOOL_WEBHOOK_SECRET is not set");

  return call(`/v1/convai/agents/${agentId}`, {
    method: "PATCH",
    body: JSON.stringify({
      conversation_config: {
        agent: {
          // Our orchestrator owns reasoning, memory, and every account action.
          // The prompt here only carries the case id through to our endpoint.
          prompt: {
            prompt: `case:${caseId}`,
            llm: "custom-llm",
            custom_llm: {
              url: `${baseUrl}/api/llm`,
              model_id: "cx-agent",
              // The tunnel is public; this is what stops anyone who finds the
              // URL from driving the agent.
              request_headers: { "x-cx-secret": secret },
            },
          },
        },
      },
    }),
  });
}
