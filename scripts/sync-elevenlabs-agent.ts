import "dotenv/config";
import { syncAgentToCustomLLM } from "../src/lib/elevenlabs";

/**
 * Points your ElevenLabs Conversational AI agent at this deployment as its
 * custom LLM. Run it after starting a tunnel, with PUBLIC_BASE_URL set to the
 * tunnel's https URL.
 */
const base = process.env.PUBLIC_BASE_URL;
if (!base || base.includes("your-tunnel")) {
  console.error("Set PUBLIC_BASE_URL in .env to your public https URL first.");
  process.exit(1);
}

const result = await syncAgentToCustomLLM(base);
console.log(`Agent now calls ${base}/api/llm/chat/completions`);
console.log(JSON.stringify(result, null, 2));
