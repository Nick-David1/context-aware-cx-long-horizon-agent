import "dotenv/config";
import { syncAgentToCustomLLM } from "../src/lib/elevenlabs";

/**
 * Re-points the ElevenLabs agent at the current PUBLIC_BASE_URL.
 *
 * Run after every tunnel restart — a cloudflared quick tunnel gets a new
 * hostname each time, and the agent stores the URL, so a stale one means the
 * call connects and the agent then says nothing.
 *
 *   npm run agent:sync                  # {{caseId}}, populated by outbound calls
 *   npm run agent:sync CASE-DEMO0002    # pin one case, for widget testing
 */
const base = process.env.PUBLIC_BASE_URL;
if (!base || base.includes("your-tunnel")) {
  console.error("Set PUBLIC_BASE_URL in .env to your public https URL first.");
  process.exit(1);
}

const caseId = process.argv[2];
await syncAgentToCustomLLM(base, caseId);

console.log(`agent → ${base}/api/llm/chat/completions`);
console.log(`case   → ${caseId ?? "{{caseId}} (injected per outbound call)"}`);
