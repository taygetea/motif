# Where the 2026-07-16 session left off (written 2026-07-19)

Cold-start handoff for the next working session. ROADMAP.md carries
direction; this carries the session-level state that would otherwise
die with the conversation.

## Landed on main, all pushed (26c0579..08159fd, 16 commits, one morning)

1. **Call-lifecycle protocol** — CallStarted/CallChunk/CallCompleted/
   CallFailed with per-call identity, retained input Msg, legacy
   five-tuple via adapter, record.py projection (node id == call_id),
   B2 eliminated. Bare verbs and `flow.group` loops are first-class in
   graph and narrate.
2. **Observer session-scoping** — registration scope-local to
   graph.session(), emission additive to process globals.
3. **Adversarial review round** (Sol xhigh + a 59-agent workflow; 18
   findings → 14 confirmed) — BaseException settlement everywhere
   (cancellation used to leak nodes/record state), gather→TaskGroup in
   best_of/tree/tournament/blackboard (ExceptionGroup to callers),
   observers can't mutate requests (params deep-copied), billed
   failures carry usage, observe()/observe_calls() validate signatures
   at attach, sessions suspend the outer current node, narrate
   surfaces descendant errors through hidden/collapsed parents.
4. **Delegated endpoints core** — DelegatedEndpoint.codex() serving
   complete/extract as single hermetic calls; requires= preflight;
   typed Attachments (transcript, flat prompt, attestation) on events
   and nodes; schema ladder (strictify → schema-in-prompt); admission
   control; subscription usage at $0 marginal. Live-verified with real
   web search on both verbs. Tests run a fake CLI that asserts the
   hermetic argv from inside.
5. Smaller: flow.group; extract() truncation → Truncated (C2);
   CostTracker exact-or-suffix matching + chunk fix + native lifecycle
   consumption (C1+); pattern boundary guards; docs sweeps throughout.

Suite: 171 → 262 tests. Full-pipeline dogfood (deep_research_v2,
$0.18) ran clean over the new seam mid-session.

## Key evaluation result (drives the next block)

Sol audited the deep_research_v2 output against a single-shot baseline
(same model tier, same question — tidally locked planet climates):
**overall 1.6× value; literature discovery 2.5×; coherent report 0.9×;
factual reliability 1.0×** (material-error rates ~20% BOTH ways — the
failure mode is "real paper, fabricated claim", worst case the
Taniguchi "hysteresis trap" the synthesis invented around a real
paper). The critique layer itself fabricated methodological objections
because critics ran from priors, without search. Full text was in
sol's state files (since overwritten); the verdict numbers and the
architectural conclusion survive here and in the ROADMAP v3 entry.
Artifacts: examples/output/research-how-do-tidally-locked-*.md (the
dossier) and examples/output/baseline-tidal-locking-singleshot.md
(the counterfactual, untracked).

## Queued, in priority order

1. **deep_research_v3 + the measurement** — investigations return
   structured claims (claim / source_url / locator / evidence_excerpt)
   via `extract(..., requires={"web_search"})` on delegated searchers;
   deterministic evidence-INTEGRITY checks (URL resolution, DOI/arXiv
   identity, excerpt occurrence — code, never semantic truth); targeted
   repair; critique-with-search on delegated critics over claim IDs +
   verifier statuses; synthesis cites claim IDs. THE MEASUREMENT: rerun
   the tidal-locking critique stage with searched critics and score the
   fabricated-objection rate against Sol's audit (its five confirmed
   critic fabrications are the ledger). Also: raise the synthesizer's
   600–1000 word cap (deep_research_v2.py:313) — it throws away most of
   what the run buys.
2. **Tool-input immutability** (ROADMAP Scale 1, S1's ghost) —
   ToolCall.input is a mutable dict inside a shallowly-frozen Msg;
   retained records are rewritable in place. Layer 1 surgery
   (render() needs plain dicts for json), property tests, pair with
   Sol, fresh session.
3. **Loom recorder** — now buildable as a plain observe_calls()
   observer (per-call identity + retained Msgs landed). Standing
   appetite; first Scale-3 artifact.
4. Smaller opens: hermetic `claude -p` adapter (design exists —
   dedicated CLAUDE_CONFIG_DIR, --strict-mcp-config, low default
   concurrency; the norms question is Olivia's); quota-window snapshot
   parsing (not in exec JSONL — maybe session files); concurrency
   behavior at N>3 (canary only proved N=3 parallel); parent-repo
   scene phase 2 (engine emits flow.group per turn, renders via
   narrate); rehearsal artifact 1 (eager preflight validation —
   requires= was the first instance of the pattern); motif-llm 0.2 to
   PyPI once the above settles.

## Standing method notes

- Sol collaboration: consult read-only, `-w` supervised for scoped
  tasks, co-author trailers on commits it contributed to; its verdicts
  get counter-analysis before transcription (symmetric adversarialism);
  terminate exchanges by state, not cadence.
- Canary-first for any new harness capability: prove the effect
  ceiling empirically before building on it.
- Never weaken the hermetic argv in delegated.py; recording granularity
  follows effect radius.
