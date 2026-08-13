import "dotenv/config";

/**
 * Runs the follow-up heartbeat against a running dev server.
 * Pass --dry to see what it would do without placing real calls.
 */
const base = process.env.PUBLIC_BASE_URL ?? "http://localhost:3000";
const dry = process.argv.includes("--dry");
const upcoming = process.argv.includes("--upcoming");

const qs = [dry && "dryRun=true", upcoming && "upcoming=true"].filter(Boolean).join("&");
const res = await fetch(`${base}/api/tick${qs ? `?${qs}` : ""}`, { method: "POST" });
console.log(JSON.stringify(await res.json(), null, 2));
