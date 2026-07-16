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
semantics; the top two landed 2026-07-16, see the ledger):

- **Deep immutability for tool inputs** (S1's ghost, promoted after the
  2026-07-16 review round). `ToolCall.input` is a mutable dict inside a
  shallowly-frozen Msg: anyone holding a Msg — or the retained
  `Node.msg` record — can rewrite recorded history in place (Sol's
  repro: `node.msg.segments[0].input["q"] = ...` changed both the
  record and the caller's original). Deep-copying retained Msgs was
  rejected: it destroys the structural sharing agent loops depend on
  (20 steps would retain 20 full history copies). The right fix is
  deep immutability at construction in prompt.py — an immutable-mapping
  freeze of tool inputs — which touches render() (json needs plain
  dicts) and the algebra tests. Layer 1 surgery; do it fresh, with
  property tests, not at the end of a long day.
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
       maxItems". Design (argued to convergence Claude↔Sol 2026-07-13,
       five rounds, both positions updated — see the labeling invariant
       at the end, it is the load-bearing part):
       - *Witness scenarios, honestly labeled*: plan-mode verbs return
         concrete schema-derived witnesses, so branch/fan/reduce/
         best_of/blackboard/tournament run their EXISTING code over
         them, spending nothing. Run host code at BOTH minItems and
         maxItems witnesses (bi-extremal scenarios; later: enum members,
         boundary values — same epistemic status). Output is "maximum
         observed across N named scenarios" — NEVER "ceiling", "bound",
         or "worst case". Counterexample forbidding the stronger claim:
         host code `if len(items) >= 5: cheap else: fan(100, expensive)`
         — the maxItems witness takes the cheap branch while a legal
         1-item result takes the expensive one. Concretization RESOLVES
         what honesty must mark OPAQUE; len() erases the dependency.
         Only locally-proven-monotone fragments may ever contribute to
         a claimed bound; values escaping such a fragment into host
         code need abstract provenance or the scenario label.
       - *Transfer functions for the semantically-branching three*:
         cascade (all-stages path), tree (depth × branching recurrence),
         agent — which needs a declared per-turn tool-call bound
         (max_steps caps nothing: compaction, finalize, tool fan-out
         escape it), and compaction breaks cost monotonicity outright
         (past the threshold, a LARGER history is CHEAPER).
       - *Two boundaries, both real*: ownership (where plan semantics
         can be installed) and semantic (where they're needed). Caveat:
         callbacks perforate ownership — fan.fn, judge_fn, filter_fn,
         tool handlers. Motif owns the invocation site, not the callback
         semantics: intercept ≠ understand.
       - *Preconditions*: plan runs create PlanNodes only, never
         enter_node(), and must not emit to module-global observers —
         cost trackers would bill imaginary calls. Session-scoping
         landed 2026-07-16 with deliberately ADDITIVE global emission
         (a startup CostTracker must not go blind when runs adopt
         sessions), so plan mode still needs an explicit mute-globals
         gate — a session option or an emission guard, designed with
         PlanNodes, not before. Also: envelopes need author-facing
         maxItems — deep_research_v2's schemas currently declare none,
         so its honest pre-run answer today is "unknown".
  The plan is a distinct representation, not fake execution:
  `Run(plan, execution)` with type-distinct PlanNode/Node joined by
  `realizes=` edges, rendered by one fold over the tagged union.
  Epistemic status ("did this happen") is never encoded as operation
  kind; no fake completed call nodes, ever — that would collide with
  both "nothing mocked" and the loom's "recorded history is what ran."
- **Delegated endpoints** (designed AND core landed 2026-07-16, Olivia +
  Claude + Sol — contract in notes/2026-07-16-delegated-endpoints-sol.md,
  canary in notes/2026-07-16-canary-results.md). LANDED: DelegatedEndpoint
  + hermetic codex adapter (delegated.py), requires= preflight on
  complete/extract, typed Attachments on events and nodes (transcript,
  flat prompt, attestation — on failures too), schema ladder
  (strictify → schema-in-prompt), per-endpoint admission control,
  subscription usage normalization, live-verified with real search.
  REMAINING: INVESTIGATION_SCHEMA + deterministic evidence-integrity
  checks + deep_research_v3 (epistemics-orchestration); a hermetic
  claude -p adapter when wanted; quota-snapshot parsing (not observed
  in exec JSONL — may live in session files). CLI harnesses (codex exec; claude -p behind a
  hermetic profile) as a third transport: subscription-priced
  intelligence with free built-in web search, versus Exa's per-call
  dollars. The converged shape:
    - No new verb, no harness noun in pipeline code: `complete`/
      `extract` over a `DelegatedEndpoint`, chosen by profile. A new
      `requires={"web_read"}` preflight makes a capability-less binding
      fail before spending instead of silently answering unresearched.
    - `flow.agent` NEVER routes to a harness — it owns handlers,
      compaction, signals. Substrate-by-profile applies to calls, not
      to motif's loop.
    - **One semantic call node per author operation under every
      profile.** Recording granularity follows effect radius, not step
      count: a sandbox-attested read-only delegation is "a very long
      LLM call with a weird thinking process" (Olivia's framing —
      reasoning-CoT and server-side search already record this way);
      write-capable delegation must be agent-shaped. The current
      anthropic-vs-deepseek searcher shape difference is
      representational debt, not precedent: if topology varies by
      profile, cross-profile run diffs are garbage.
    - Contract essentials: effect attestation emitted by the runner
      (recorder preserves, never trusts flags), statelessness
      (--ephemeral / --no-session-persistence), fine-grained
      capabilities (browser_interact is a WRITE), typed Attachments on
      CallCompleted AND CallFailed (transcript forensics matter most on
      failures — never overload Node.msg), invocation fingerprint,
      usage with usage_complete/billing_basis/quota snapshot (zero
      dollars ≠ zero resource), normalized failure taxonomy. No turn
      bound — unenforceable; the honest envelope is "interior opaque,
      wall-time ≤ T".
    - Breaks-first (Sol, partly empirical on codex 0.144.3): no tool
      allowlist flag exists, so EXTERNAL_READ is unproven until a
      capability canary passes (canary BEFORE any motif code); the
      controller needs writable runtime separate from the model's
      read-only sandbox; clean mode may disable the search capability
      the lane exists for; parallel-session quota behavior on one
      subscription is unmeasured; JSONL is an adapter protocol —
      pin versions. Subscription-bulk-use policy is deployment
      configuration, never a motif invariant.
    - Downstream: deep_research_v3 becomes epistemics-orchestration —
      decompose → delegated structured investigations (claim/URL/
      locator/excerpt at research time; transcripts are forensics, not
      evidence) → deterministic evidence-INTEGRITY checks (code:
      resolution, identity, excerpt-occurrence — never semantic truth)
      → targeted repair → delegated critique over claim IDs +
      verifier statuses → synthesis that cites claim IDs. This is the
      structural answer to the 2026-07-16 evaluation finding that the
      critique layer fabricates methodology from priors.
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
- Remaining test gaps from the 2026-07-08 review: show components, TUI
  internals, the anthropic streaming path (the SDK fake has no
  `.stream()`).
- TUI reads the salience policy (it currently has its own display logic;
  2026-07-16 note: NodePanel now reads streamed output through to
  llm_call record children — a stopgap, not the unification).
- **motif-llm 0.2 to PyPI** once the above settles.
- Design decisions from the 2026-07-16 review round, recorded so they
  stay decisions rather than defaults:
    - **Additive global observers stand — for now.** Sol argued
      exclusive-by-default (tenant confidentiality: ambient process
      observers see every session's CallStarted.msg; double-delivery
      when the same tracker is attached globally and locally;
      clear_observers() is context-sensitive ambient behavior). The
      counter that carried today: motif's current population is single-
      experimenter, and exclusive semantics silently blind a startup
      CostTracker the moment runs adopt sessions — a worse invisible
      mistake than any of the above at present scale. REVISIT BEFORE
      ANY MULTI-TENANT DEPLOYMENT: likely session(isolate=True) or a
      payload-minimized event stream for global telemetry.
    - **A session is a scope, not a lifetime barrier.** Tasks spawned
      inside a session that outlive its block keep emitting to it —
      s.roots can grow after narrate() starts reading. Documented, not
      prevented; a cancellation boundary is Scale-2 runtime work.
    - **Abandoned async generators settle in the GC-finalizer's
      context**: an unclosed stream's CallFailed may miss session-
      scoped observers (globals and the projection always see it).
      Close what you abandon (`aclosing`); noted in stream()'s code.
    - The graph record is not yet a full replay record (params/author
      meta/endpoint config not all stamped; to_dict drops msg) — that
      is Scale-2 replay work, tracked there, and "the complete record"
      language was softened accordingly.
- Resolved 2026-07-16 (the call-identity day; fixes by Claude, test-gap
  audit by Sol):
    - **The review-round fixes** (evening; Sol xhigh audit + a 59-agent
      adversarial workflow, 18 raw findings → 14 confirmed →
      deduplicated): BaseException settlement everywhere (cancellation
      leaked record._open and running-forever nodes — fan's own
      TaskGroup triggered it first-party); gather → TaskGroup in
      best_of/tree/tournament/blackboard (failing calls now cancel
      siblings instead of billing on; callers see ExceptionGroup);
      CallStarted deep-copies schema/tools/extra (observers could
      mutate the outgoing request through the event); usage captured
      before parsing so CallFailed pays its bill, and CostTracker
      consumes lifecycle events natively; observe()/observe_calls()
      signature validation at attach (the mix-up was silently
      swallowed); exit_node error is `is not None` + never-empty error
      text (empty-message exceptions recorded as success); sessions
      suspend the outer current node; narrate surfaces descendant
      errors through hidden/collapsed parents; flow.call(schema=) no
      longer replaces data with a preview; full-length call ids; TUI
      reads the first record and dispatches the new kinds.
    - **Call-lifecycle protocol landed.** CallStarted/CallChunk/
      CallCompleted/CallFailed with per-call identity, retained input
      Msg, and usage-on-failure; legacy five-tuple derived by adapter;
      record.py projects events into llm_call graph nodes (node id ==
      call_id); B2 eliminated (llm.py no longer imports graph); bare
      verbs and raw loops are first-class in the graph and in narrate;
      fan children became honest item slots; flow.call() became an
      annotation; abandoned streams settle as CallFailed.
    - **Observer session-scoping landed.** Registration is scope-local
      to graph.session(), emission additive to process globals; flow
      and show registries included; clear_observers() clears only the
      active scope and can never silence the projection.
    - `flow.group(title)` — the raw-loop story completed.
    - `extract()` truncation raises Truncated with the raw partial
      (C2 closed); CostTracker exact-or-suffix matching, chunks no
      longer counted as calls (C1 closed).
    - Pattern guards: best_of([]) fails at the boundary; tournament
      validates the judge's verdict is literally 'a'/'b'; tournament
      rounds, blackboard rounds, and cascade attempts error their node
      instead of staying "running" forever.
    - Test gaps closed for best_of, tournament, flow.call, label_key.
- Resolved 2026-07-13 (cross-model audit by GPT-5.6 Sol via codex, fixes
  by Sol + Claude): tool-input mutation could rewrite recorded history
  (S1); `tree()` accepted invalid paragraph partitions silently (S2);
  `complete()` swallowed max_tokens truncation (S3); role profiles were a
  process global (S4); `branch()` guessed the topology from response
  order (S5). Pattern worth keeping: every fix validates before the money
  is spent and puts the fix in the error message.

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
       (Satisfied 2026-07-16: the call-lifecycle protocol gives every
       call its own identity and event stream — identical resamples are
       distinct facts. A loom recorder is now buildable as a plain
       observe_calls() observer.)
    3. Never memoize/dedupe calls by content — caching identical calls
       would silently collapse deliberate resamples into one node.
    4. The graph stays open to additive edge types (variant_of siblings).
       (Call nodes retaining their input Msg — the gap this item named —
       closed 2026-07-16: CallStarted.msg / Node.msg. Trace
       serialization of retained Msgs is the remaining open design
       question; to_dict() deliberately skips them.)
- **Counterfactual debugging** (the loom's human-facing verb; replaces
  both "blame walk" and pipeline-optimization framings, which were
  considered and rejected 2026-07-13 — human judgment is too sparse to
  train on, and today's graph records containment, not dataflow). The
  honest version: mark an output span → see the emitting call plus its
  likely producers → fork one candidate with identity-addressed
  upstream reuse → replay downstream → compare. A sparse judgment
  becomes an experimental query, not a training signal. Two-sided
  refinement (converged 2026-07-13): now that call-lifecycle events
  retain input Msgs (landed 2026-07-16), **lexical lineage** is
  recoverable post hoc — motif
  pipelines interpolate upstream outputs into downstream prompts
  near-verbatim (verified across deep_research_v2's whole chain), so
  producer edges fall out of span-matching, yielding an evidence-ranked
  "producer evidence" set. Known degradations: short/boilerplate spans
  (false positives), host transformations (false negatives), compaction
  summaries (edges survive, span ancestry doesn't), identical resamples
  (indistinguishable by content — which is precisely why call identity
  is irreplaceable), and dense fan-in (the synthesis call consumes
  every brief, so matching adds nothing exactly where credit matters
  most). Store spans, offsets, uniqueness, ordering. Never call it
  blame: matching ranks candidates; only counterfactual replay tests
  responsibility. Before loom replay exists, this is just drill-in
  navigation and should not be built as a separate feature.
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
