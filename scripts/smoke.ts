import "dotenv/config";
import { recall, rerankPath } from "../src/lib/memory";
import { config } from "../src/lib/config";

/**
 * Exercises the retrieval path end to end against real data.
 *
 * `verify` proves the credentials and index exist; this proves the pipeline
 * actually returns sensible context — which is the thing that breaks silently.
 */

const queries = [
  { q: "I need to move my payment to the 15th", customerId: "CUST-1001" },
  { q: "why did my payment get missed, money is tight", customerId: "CUST-1002" },
  { q: "another bank offered me a better rate", customerId: "CUST-1003" },
];

console.log(`\nembedding=${config.embeddingMode}  rerank=${rerankPath()}\n`);

for (const { q, customerId } of queries) {
  console.log(`── "${q}"  (${customerId})`);
  const hits = await recall({ query: q, customerId });

  if (hits.length === 0) {
    console.log("   no results — index may still be building, or seed didn't run\n");
    continue;
  }

  for (const h of hits) {
    const rr = h.rerankScore !== undefined ? ` rr=${h.rerankScore.toFixed(3)}` : "";
    const win = h.winRate !== undefined ? ` win=${(h.winRate * 100).toFixed(0)}%` : "";
    console.log(
      `   ${h.kind.padEnd(8)} ${h.score.toFixed(3)}${rr}${win}  ${h.text.slice(0, 88)}`,
    );
  }
  console.log("");
}

process.exit(0);
