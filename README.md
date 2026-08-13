# Context-Aware CX — Long-Horizon Agent

A long-horizon customer service agent that learns over time.

Most AI support agents handle one conversation and forget it. This one works a **case** — "get this loan refinanced", "collect this past-due balance" — across days, calls, and channels, and gets better at that class of case every time one closes.

Built on **MongoDB Atlas** (structured case state + Vector Search context engine), **Claude** (reasoning, planning, reflection), and **ElevenLabs Conversational AI** (voice).

---

## The idea

Two things are missing from most LLM support agents:

**Local context.** The policies, edge cases, and tribal knowledge that define how a specific business actually operates. RAG injects snippets at inference time and throws them away. This system keeps a durable, structured context layer in Atlas and retrieves from three slices of it on every turn:

| Slice | What it is | Scope |
|---|---|---|
| `policy` | Authored business logic. Supersedable, so context stays current instead of just growing. | Global |
| `episodic` | What actually happened with this specific customer. | Per-customer |
| `lesson` | Generalizations the agent derived from its own outcomes. | Global |

They're three separate vector searches rather than one top-k, because a chatty customer's notes will otherwise crowd out the policy that governs the action.

Retrieval is two-stage. `$vectorSearch` over-fetches ~4× the quota per slice, then **`rerank-2.5`** — a cross-encoder — scores each candidate jointly against the query, and the quota is applied to *that* order. Vector search embeds query and document separately, which is what makes it fast over the whole collection and blunt at the top of the list; the cross-encoder reads the pair together and is the opposite. You can watch it work in the smoke test: for *"why did my payment get missed, money is tight"* the hardship-disclosure policy sits at vector rank #2 and the reranker promotes it to #1, above the generic hardship-plan description that scored higher on vocabulary overlap.

**Both stages run inside MongoDB by default.** The vector index is declared `type: "autoEmbed"`, so Atlas generates embeddings from the `text` field on write *and* vectorizes the query string at read time — `$vectorSearch` receives `query`, not `queryVector`, and no vector ever crosses the wire. Reranking is the `$rerank` aggregation stage. MongoDB acquired Voyage AI in Feb 2025, so store, index, embed, and rerank are all one system.

Two escape hatches, because neither is universally available:

| Env | Default | Fallback |
|---|---|---|
| `EMBEDDING_MODE` | `auto` — Atlas embeds in-database, no key needed | `client` — we call the API and store vectors |
| `RERANK_MODE` | `native` — `$rerank` stage, needs MongoDB 8.3 + Native Reranking enabled | `client` — we call the rerank API; `off` |

`native` downgrades to `client` automatically the first time a cluster rejects the stage, so an 8.1 cluster still works without config changes.

**A learning loop.** When a case closes:

1. Every lesson that was in context gets its win/loss counters incremented — credit assignment.
2. A reflection pass reads the full transcript plus the outcome and writes new lessons back into the store.
3. Retrieval prefers lessons with a track record, and drops ones that lose more than they win.

The result compounds: the hundredth refinance case is worked better than the first, and the improvement lives in your database, not in a model checkpoint.

**No cold start, at two timescales.** Across cases, that's the memory layer above. Within a single turn, it's checkpointing: the agent's tool loop runs against live servicing systems, and every completed step is written to Atlas before the next one begins. If the process dies mid-refinance, the next turn resumes from the last completed tool call — it doesn't replay the side effects, and it doesn't start the case over.

## Architecture

```
                    ┌──────────────────────────────┐
   phone call ──────│  ElevenLabs Conversational AI │  STT · turn-taking · TTS
                    └───────────────┬──────────────┘
                                    │  custom LLM (OpenAI-compatible)
                    ┌───────────────▼──────────────┐
   dashboard chat ──│    Case orchestrator         │  Claude + tool loop
                    └───┬──────────────────────┬───┘
                        │                      │
              ┌─────────▼────────┐   ┌─────────▼─────────┐
              │  Context engine  │   │  Servicing actions │
              │  $vectorSearch   │   │  validate → execute│
              │  → rerank-2.5    │   │  (rules in code)   │
              └─────────┬────────┘   └─────────┬─────────┘
                        │                      │
                    ┌───▼──────────────────────▼───┐
                    │        MongoDB Atlas          │
                    │ cases · loans · interactions  │
                    │ memories · outcomes           │
                    │ checkpoints (resume state)    │
                    └───────────────────────────────┘
```

ElevenLabs is wired in as the agent's **custom LLM**, not via per-tool webhooks. It owns voice; the orchestrator owns reasoning, memory, and every account action. The phone call and the dashboard chat run through the exact same orchestrator, so there's no drift between channels.

**Write actions are gated by deterministic code, not by the prompt.** The model *proposes* an action; `src/lib/actions.ts` decides whether it's allowed. A hallucinated proposal fails validation instead of moving money.

## Setup

```bash
npm install
cp .env.example .env      # MONGODB_URI and ANTHROPIC_API_KEY are the only required values
npm run db:indexes        # creates the autoEmbed vector index (~1 min to build)
npm run verify            # confirms every credential and the index status
npm run db:seed           # 3 customers, 3 open cases, policy corpus, seeded lessons
npm run smoke             # proves retrieval returns sensible context
npm run dev
```

Open http://localhost:3000.

You need an **Atlas** cluster — M0 free tier is fine. Vector Search doesn't exist on a local `mongod`, so `db:indexes` will fail against localhost.

In the default `auto`/`native` modes **no model API key is required** — Atlas holds it. Set `VOYAGE_API_KEY` (Atlas UI → **AI Model APIs → Create model API key**) only if you switch either mode to `client`. Note there are two incompatible surfaces: an Atlas model key works against `ai.mongodb.com` and a voyageai.com key against `api.voyageai.com`; crossing them returns 403.

`verify` checks each dependency independently and `smoke` exercises the real retrieval path, so a failure names the thing that's actually wrong instead of surfacing three layers deep at demo time.

## Demo script

**1. Context that a generic model can't have.** Open `CASE-DEMO0001` (Dana, payment date change) and type:

> I need to move my payment to the 15th

Watch the right-hand panel: policy on the 90-day rule, the customer's own history, and a lesson with a 88% win rate all get retrieved. The agent verifies identity before touching anything (DOB `1988-04-12`, SSN `4417`).

**2. Eligibility that can't be talked around.** Open `CASE-DEMO0002` (Marcus, past due) and ask to move *his* payment date. The loan is $412 past due, so `check_eligibility` blocks it and the agent explains what would make him eligible — instead of promising something the servicing system will reject.

**3. Long horizon.** That case has a follow-up already due. Run:

```bash
npm run tick -- --dry
```

It finds the case, knows why it's calling back ("he asked to be called after payday on the 15th"), and — with ElevenLabs configured — places the call.

**4. Learning.** Work a case to a conclusion. On `close_case`, reflection reads the transcript and writes new lessons into Atlas. Re-run the same scenario with a different customer and those lessons show up in the retrieved-context panel.

## Voice

```bash
npx ngrok http 3000            # or any tunnel
# set PUBLIC_BASE_URL in .env to the https URL
npm run agent:sync             # points your ElevenLabs agent at /api/llm
```

Create the agent in the ElevenLabs dashboard first and put its id in `ELEVENLABS_AGENT_ID`. For outbound calls, connect a phone number and set `ELEVENLABS_PHONE_NUMBER_ID`. The sync script sets the agent's prompt to `case:{{caseId}}` — that's how the case id reaches the orchestrator, since ElevenLabs passes dynamic variables through the prompt.

## Layout

```
src/lib/
  types.ts        domain model
  mongo.ts        Atlas client + collection handles
  config.ts       embedding/rerank modes, model ids, base URLs
  embeddings.ts   client-side embed + rerank (only used outside auto/native mode)
  memory.ts       the context engine: recall, remember, supersede, score
  actions.ts      servicing actions + the eligibility rules that gate them
  agent.ts        Claude tool loop + the reflection pass
  elevenlabs.ts   outbound calls + agent config sync
src/app/api/
  llm/chat/completions/   OpenAI-compatible endpoint ElevenLabs calls
  chat/                   text channel (dashboard)
  cases/                  case queue + detail
  tick/                   long-horizon follow-up heartbeat
  voice/outbound/         place a call for a case
  memories/               browse or vector-search the context engine
scripts/
  create-indexes.ts  db:indexes (autoEmbed or client vector index)
  verify.ts          verify
  smoke.ts           smoke
  seed.ts            db:seed
  tick.ts            tick
  sync-elevenlabs-agent.ts  agent:sync
```

## Notes

- The dashboard is an inspector, not the product. It exists so you can watch which memories were retrieved and which actions fired; the agent runs headless on the phone.
- `nextActionAt` on a case is what makes this long-horizon rather than session-scoped. The agent sets it itself via `set_plan`, with a reason.
- `checkpoints` holds at most one `in_flight` document per case. `runAgentTurn` looks for it before doing anything else and resumes from `step` if it finds one.
- Reflection is fired and not awaited when a case closes — a slow model call never blocks the customer's turn.
- `active: false` on a memory is how context stays current. Superseded policy is excluded at the vector-search pre-filter, so stale rules can't be retrieved at all.
