# Roadmap

## The image (held loosely)

> "The fake sci-fi interface to an AI doing something complicated — but
> it's real."

The movie computer: panes of live activity arranged around a central
process, everything moving, everything legible at a glance. Hollywood fakes
it because real computation was never legible enough to film. The bet here
is that LLM computation — structured as an explicit graph, with declared
roles and derived labels — actually is. Every pane a true view of running
work; every label load-bearing; nothing mocked.

Concretely: a local web interface connected to a Claude session with motif.
You describe an experiment. The session authors a pipeline (visible — the
plan, the code, the fix-loop are part of the run). The interface assembles
itself from the pipeline's shape before a token is spent. You approve the
shape and the estimated cost. Then you watch it run.

## What it's for

An **instrument**, not an automation tool. The recurring workload is
studying the behavior of LLM systems themselves: synthetic nervous systems
under pressure, research strategies escaping their framing, swarms
converging or failing to. Watching is not the reward at the end — it is the
point. Instruments demand things dashboards don't: replay, comparison
across runs, steering, and salience control at read time.

## Principles (nailed to the wall)

1. **The author has no mistakes available to make.** Motif's reason to
   exist. Every API change is tested against: *what mistake does this make
   possible?* The author is a model writing pipelines fresh, every time —
   the library is the prompt.
2. **Replay is free.** The live view and the saved trace are one structure
   at two moments. If replaying a trace doesn't reproduce the live
   rendering, we built two display systems.
3. **Reader > author > policy.** Salience defaults come from the policy,
   the pipeline author can override them, and the person watching outranks
   both — drill-in is always available, whatever the layout.
4. **Labels from data, not diligence.** `label_key=` over `title=`
   wherever a label can be derived; author labels are the fallback. Author
   diligence does not survive autonomous authoring.
5. **Rehearse the shape before spending.** A mocked pass builds the graph
   skeleton in milliseconds: the empty interface, the call count, the cost
   estimate, and the author's type-checker (unbound roles, bad schemas,
   wrong kwargs) — all before money.
6. **Display quality never depends on the blessed patterns.** Raw loops
   over the verbs must trace and render as well as `branch → fan → reduce`,
   or the instrument can only see experiments it already knows.
7. **Glitches are fixed in the renderer, never per-pipeline.** The model
   authors declarations; the renderer is fixed, tested infrastructure.
   Model-generated frontend code is how the dream rots.

## Scale 1 — the library (weeks)

The remaining error surface, in rough order of leverage (reordered
2026-07-13 after a Claude/Sol design conversation — call identity moved
above rehearsal so replay and planning don't harden around weaker
semantics):

- **Call-lifecycle protocol.** Per-call identity at the verb boundary:
  `CallStarted(call_id, verb, msg, declared_role, resolved_endpoint,
  params)` / `CallChunk` / `CallCompleted(result, usage)` / `CallFailed` —
  pure Layer-2 facts emitted through the observation seam (a new
  signature; an adapter may preserve the legacy five-tuple, but don't
  promise compatibility). A graph-side projection folds them into
  execution nodes: bare verbs get automatic call nodes (fixing the
  standing principle-6 violation that raw verb loops are invisible to
  the tree), `llm.stream()`'s direct graph mutation (exception B2) is
  eliminated rather than grown, and identical concurrent resamples
  become distinguishable — today `best_of` judgments, tournament
  matches, and tree's split/leaf/merge calls all share one node id.
  Migration notes: fan's preallocated children become honest item slots
  the call records link beneath; `flow.call()` becomes an author
  annotation around an automatically recorded call, not a second claim
  that a call occurred.
- **Observer session-scoping.** `llm._observers` / `flow._observers` /
  `show._show_observers` are the last shared globals (graph roots are
  session-scoped; the role profile became a ContextVar in the 2026-07-13
  fixes — it was a shared global too, which this entry wrongly denied).
  This is the Runtime seam the README names.
- **Rehearsal, split into three honest artifacts** (was "shape
  rehearsal" — the single-feature framing hid the hard question of
  data-dependent topology):
    1. *Eager preflight validation* in ordinary execution, not a mode:
       roles bound, schemas well-formed, kwargs rejected, tool schemas
       matched to handlers — at verb entry, before the request. (The
       2026-07-13 branch() fix is the pattern.)
    2. *Scenario dry-run*: no-transport verbs, reports which call sites
       one control-flow scenario actually reached. Named honestly — a
       mocked branch() result selects one downstream path; this is not
       exhaustive validation.
    3. *Topology envelope*: exact nodes, bounded repetitions (schema
       minItems/maxItems, max_steps), and opaque expansion sites shown
       as unknowns — "fan: 2–8 branches", "agent: ≤20 turns". Cost is
       an interval or `unknown`, never a scalar estimate; an unbounded
       cost-affecting cardinality reports "cost ceiling unknown; add
       maxItems".
  The plan is a distinct representation, not fake execution:
  `Run(plan, execution)` with type-distinct PlanNode/Node joined by
  `realizes=` edges, rendered by one fold over the tagged union.
  Epistemic status ("did this happen") is never encoded as operation
  kind; no fake completed call nodes, ever — that would collide with
  both "nothing mocked" and the loom's "recorded history is what ran."
- **`flow.group(title)`.** A grouping node with no LLM call — turns,
  phases, authoring sections. Tiny, worth doing early, but not
  architecturally prior to the above.
- **Schema ergonomics.** Hand-written JSON Schema dicts are the largest
  remaining author error surface by volume. Decide: rehearsal-side
  validation (no new API) vs. a schema helper (new surface). Lean
  validation-first.
- **Msg provider-validity checks** (reframed from "enforce the nested-Msg
  convention"): validate that a Msg is well-formed for the transport —
  tool_use/tool_result pairs closed, roles renderable — but do NOT brand
  Msgs with scope ownership or prohibit reuse. Feeding a recorded prefix
  into alternative continuations is the loom's core operation; a
  provenance rule would preserve today's convention by precluding
  tomorrow's central feature.
- Open findings from the 2026-07-08 review: CostTracker prefix-matching
  false positives (C1); `extract()` truncation legibility (C2); test gaps
  for `best_of`, `tournament`, `flow.call`, `label_key`, show components,
  TUI internals. (C2 note: `complete()`/`stream()` truncation now raises
  `Truncated` as of 2026-07-13; `extract()` remains the open half.)
- Resolved 2026-07-13 (cross-model audit by GPT-5.6 Sol via codex, fixes
  by Sol + Claude): tool-input mutation could rewrite recorded history
  (S1); `tree()` accepted invalid paragraph partitions silently (S2);
  `complete()` swallowed max_tokens truncation (S3); role profiles were a
  process global (S4); `branch()` guessed the topology from response
  order (S5). Pattern worth keeping: every fix validates before the money
  is spent and puts the fix in the error message.
- TUI reads the salience policy (it currently has its own display logic).
- **motif-llm 0.2 to PyPI** once the above settles.

## Scale 2 — the instrument (a season)

- **The web fold.** SSE bridge: `graph.session` → browser. A live DOM
  renderer over the *same* salience policy as `narrate` (third
  homomorphism). Three layouts to start — document (research-shaped), chat
  (scene-shaped), board (fan/swarm-shaped) — chosen from graph shape or one
  author hint. Append-stable layout: patterns that know their cardinality
  (fan stamps `count=`) pre-allocate space, so structure streams without
  reflow.
- **Replay and runs.** Run identity; trace replay through the identical
  render path; diff view over two runs of the same pipeline (the second
  thing an experimenter ever wants).
- **Retire the FlowEvent layer.** Trace/LiveFlowDisplay currently consume
  the legacy event stream; fold them onto the graph. Three display layers
  become two (graph + show), deliberately.
- **Scene phase 2** (parent repo): the engine emits graph nodes
  (`flow.call` around evaluate/regulate/speak, `group` per turn), scenes
  render through narrate/the web fold, CostTracker in `run.py`. The
  conversation instrument joins the same display stack.
- **The authoring loop.** Browser request → Claude session authors a
  pipeline → rehearsal → approve → run. The authoring session is itself a
  node in the run's graph (plan, generated code, tracebacks, revisions —
  watchable and auditable). Pipeline library: named, parameterized,
  reusable apparatus ("do we already have one like this?"). Known wrinkle:
  nested `claude` sessions don't inherit Max auth on this machine —
  `claude-agent-sdk` (already a parent-repo dependency, proven by the
  persona builder) or an API key when available.

## Scale 3 — the lab (quarters, held loosely)

- **Steering.** Pause at pattern boundaries; edit a prompt or profile;
  fork-from-node without re-paying for upstream work. Requires run
  persistence/resume — the deepest layer motif doesn't have. Design note
  (2026-07-13): profiles are now context-local, so a steering surface
  cannot rebind them from outside the run — pause points must apply
  edits from within the run's context. That is the correct shape anyway
  (the run stays the sole authority over its own bindings), but the
  control channel has to be designed for it. Until then,
  cheap-profile economics make "just rerun it" a legitimate recovery
  strategy: iterate on flash/local, promote to opus once.
- **The loom.** Steering's general form: a tree of alternative
  continuations — N samples from one prefix, each spawning its own
  downstream subtree, per-branch profiles (resample this subtree on opus,
  leave siblings on flash). Its own thing; not scheduled. What IS current
  is the do-not-preclude list — invariants any change gets held against
  so the loom stays buildable:
    1. Verbs stay pure functions over immutable Msgs; recorded history is
       what actually ran (the 2026-07-13 immutability fix is load-bearing
       here).
    2. The observation seam keeps carrying the full per-call facts.
       (Corrected 2026-07-13: the current tuple gives node-scoped
       correlation — _notify runs in-task, so current_node() is
       reachable — but NOT per-call identity; multi-call pattern nodes
       like best_of share one node, so identical resamples are
       indistinguishable. The call-lifecycle protocol in Scale 1 is the
       fix; a loom recorder becomes fully buildable once it lands.)
    3. Never memoize/dedupe calls by content — caching identical calls
       would silently collapse deliberate resamples into one node.
    4. The graph stays open to additive edge types (variant_of siblings);
       call nodes retaining their input Msg is the known additive gap.
- **Comparison as a first-class workflow.** Sweeps (persona × condition,
  prompt × model), aggregate views over many runs, the lab notebook that
  accretes: apparatus + traces + conclusions.
- **Swarm-scale display.** At N=3000 salience lives in the aggregate —
  histograms, exemplars, convergence views. Cardinality is already the
  policy's signal; grow the aggregate vocabulary.
- **The conversation instrument.** regulatedconversation experiments driven
  from the lab: regulation on/off ablations, arousal-signature sweeps,
  long-horizon scenes with compaction.
- Maybe: remote execution (gnon.moe), runs that outlive the laptop,
  `claude -p` as a pipeline-authoring backend once the auth story is clean.

## Anti-goals

- No drag-and-drop pipeline GUI. The pipeline representation is Python on
  one page — inspectable, diffable, versionable. That's a feature.
- No model-generated frontend code. Declarations in, fixed renderer out.
- No premature Runtime object. The observer seam gets an explicit runtime
  when the server needs it, not before.
