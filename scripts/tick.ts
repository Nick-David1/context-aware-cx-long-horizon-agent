import "dotenv/config";

/**
 * Runs the follow-up heartbeat against a running dev server.
 * Pass --dry to see what it would do without placing real calls.
 */
const base = process.env.PUBLIC_BASE_URL ?? "http://localhost:3000";
const dry = process.argv.includes("--dry");

const res = await fetch(`${base}/api/tick${dry ? "?dryRun=true" : ""}`, { method: "POST" });
console.log(JSON.stringify(await res.json(), null, 2));
