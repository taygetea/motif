The minimal design is:

> Keep the existing verbs. Add a delegated endpoint transport for `complete`/`extract`. Do not route `flow.agent` wholesale to a harness. Preserve one semantic call node across profiles; put substrate-internal activity in a typed transcript attachment.

Claude’s claims (a) and (b) do not both survive. Claim (b) is right. Claim (a) must be narrowed to “a delegated endpoint may internally run an agent loop.” `flow.agent` itself remains motif-owned.

## 1. What the author writes

For ordinary free-form investigation:

```python
SEARCHER = llm.role("searcher")

brief = await llm.complete(
    RESEARCHER | user(question),
    model=SEARCHER,
    requires={"web_read"},
)
```

For research intended to feed citation verification:

```python
investigation = await llm.extract(
    RESEARCHER | user(question),
    schema=INVESTIGATION_SCHEMA,
    model=SEARCHER,
    requires={"web_read"},
)
```

No `delegate()` verb. No harness noun in pipeline code. `requires=` describes the work; the profile selects the substrate.

Profiles may bind `SEARCHER` to:

```python
{
    "searcher": DelegatedEndpoint.codex(...),
}
```

or:

```python
{
    "searcher": ServerToolEndpoint(..., capabilities={"web_read"}),
}
```

or eventually:

```python
{
    "searcher": CompositeEndpoint.exa(...),
}
```

The existing [Endpoint](/home/taygetea/code/regulatedconversation/motif/src/motif/llm.py:422) currently conflates endpoint identity with two transport choices (`base_url is None` versus OpenAI-compatible). A delegated endpoint needs a real third transport type, not an `extra` convention.

### What appears in the graph

| Profile substrate | Author operation | Graph | Attachment |
|---|---|---|---|
| Codex harness | `extract(... requires={"web_read"})` | one `llm_call` | Codex JSONL transcript |
| Server-side web search | same | one `llm_call` | provider tool trace, if exposed |
| Composite Exa investigation | same | one `llm_call` | internal search-loop transcript |
| Plain model lacking web | same | preflight failure before spending | none |
| Custom motif handlers | `flow.agent(...)` | `agent → step → llm_call/tool_call...` | ordinary graph children |

That gives stable semantic graph shape across profiles. Physical execution differs, but physical trace is attached rather than projected into semantic topology.

The current difference between Anthropic server-side search and the DeepSeek/Exa loop should be treated as representational debt, not precedent. If profile substitution changes a call from one node into twenty graph nodes, then:

- rehearsal topology changes by profile;
- comparisons and replay match different causal units;
- salience/rendering changes;
- “cost is a run property, not a program property” quietly becomes “program topology is a run property.”

That is too expensive an inconsistency.

### `flow.agent` must not route to a harness

[flow.agent](/home/taygetea/code/regulatedconversation/motif/src/motif/flow.py:1258) owns handlers, compaction, signal tools, error conversion, max-step semantics, and finalization. A harness cannot execute arbitrary Python handlers. Therefore:

```python
await flow.agent(..., tools=handlers, tool_schemas=schemas)
```

always means motif’s loop. A profile may change the model serving each `act()` call, but may not replace the entire pattern with Codex or Claude.

If an author merely wants “investigate using web,” they should not write `flow.agent`; that was leaking the old Exa implementation into the program.

## 2. Minimal delegated-endpoint contract

Your proposed list is directionally right, but “turn/quota bound” is the wrong mandatory field, and effect radius needs more substance.

I would require this logical contract:

```python
DelegatedEndpoint(
    id=...,
    adapter="codex",
    model=...,

    supported_verbs={"complete", "extract"},
    capabilities={"web_read", "fs_read"},
    effect_ceiling=Effect.EXTERNAL_READ,

    stateless=True,
    timeout_seconds=...,
    max_concurrency=...,
    max_output_bytes=...,

    invocation_fingerprint=...,
    transcript_policy=...,
    usage_policy=...,
)
```

### Required

1. **Supported verbs and capabilities**

   Codex may support `complete` and schema-backed `extract`; it does not support motif `act()` with host handlers. Capability names should be narrow:

   - `web_search`
   - `web_fetch`
   - `fs_read`
   - `browser_interact`
   - `external_write`

   A single `"web"` capability is too broad. Browser interaction can submit forms or trigger external state; search/fetch should remain distinct.

2. **Effect ceiling plus enforcement attestation**

   The endpoint declares the maximum possible causal effect, but the runner must derive and record how it was enforced:

   ```python
   EffectAttestation(
       sandbox="read-only",
       allowed_tools=("web_search", "web_fetch", "read"),
       config_sources=...,
       hooks_disabled=True,
       plugins_disabled=True,
       mcp_servers=(),
       cwd=...,
   )
   ```

   The recorder should not “trust sandbox flags.” Its job is to record facts. The delegated runner validates the effective invocation and emits the attestation.

   Filesystem read-only alone is insufficient: Codex can have apps, MCP, browser/computer use, hooks, and plugins capable of external effects.

3. **Statelessness**

   Default delegated calls must be fresh and non-resumable:

   - Codex: `--ephemeral`
   - Claude: `--no-session-persistence`

   Session reuse must be a separate, explicit endpoint type or capability. Hidden cross-call state breaks replay and makes the retained input `Msg` no longer describe what the model knew.

4. **Enforceable process bounds**

   Require:

   - wall-clock deadline;
   - process-tree termination;
   - output byte limit;
   - concurrency/admission limit.

   A turn bound is not presently enforceable through installed `codex exec`. A prompt saying “use at most N turns” is not a bound. Rehearsal should report:

   > one delegated invocation; internal expansion opaque; wall time ≤ T; quota consumption unknown.

5. **Output contract**

   The adapter must separate:

   - validated final result;
   - emitted transcript;
   - stderr/diagnostics;
   - partial output on failure.

   Codex has `--output-schema`; Claude has `--json-schema`. Use these for `extract`, then independently validate returned JSON in motif.

6. **Typed, durable attachments**

   Do not overload `Node.msg`. It currently means retained input, and [is not even serialized by `to_dict()`](/home/taygetea/code/regulatedconversation/motif/src/motif/graph.py:52).

   Add something like:

   ```python
   Attachment(
       kind="harness_transcript",
       media_type="application/x-ndjson",
       uri=...,
       sha256=...,
       byte_length=...,
       complete=True,
       redaction="...",
   )
   ```

   `CallCompleted` and `CallFailed` both need attachments. Failed investigations are often where the transcript matters most.

7. **Normalized usage with provenance and completeness**

   Current `CostTracker` assumes missing fields mean zero. That is dishonest for harnesses.

   Codex transcripts expose cumulative token counts and subscription rate-limit snapshots. Normalize them as:

   ```python
   {
       "input_tokens": ...,
       "output_tokens": ...,
       "cache_read_tokens": ...,
       "usage_complete": True,
       "usage_source": "codex_jsonl",
       "billing_basis": "subscription",
       "marginal_cost_usd": 0,
       "quota": {
           "used_percent": ...,
           "window_minutes": ...,
           "resets_at": ...,
       },
   }
   ```

   “No billed dollars” is not “no resource use.” Rehearsal and admission control care about quota pressure.

8. **Normalized failure semantics**

   At minimum:

   - `auth_failed`
   - `quota_exhausted`
   - `rate_limited`
   - `deadline_exceeded`
   - `sandbox_initialization_failed`
   - `capability_unavailable`
   - `invalid_structured_output`
   - `process_failed`
   - `cancelled`

   Include retryability, reset/retry time, partial usage, exit code, and transcript attachment.

9. **Invocation fingerprint**

   Record:

   - CLI and adapter version;
   - model and effort;
   - exact safe configuration digest;
   - cwd;
   - instruction/config sources;
   - enabled capabilities;
   - schema hash;
   - rendered-prompt hash.

   The original `Node.msg` is not enough because a CLI supplies its own system prompt, repo instructions, configuration, and tools.

### Superfluous or misplaced

- **Mandatory turn bound:** unenforceable and not semantically relevant to an atomic delegated call.
- **Per-call quota bound:** mostly belongs in an endpoint admission controller. The endpoint should expose quota observations; the profile/run policy decides whether another call may begin.
- **Recorder trust of the endpoint:** wrong layer. The runner attests; the recorder preserves.
- **Transcript in `meta`:** too large, untyped, and not durable.
- **Cost reported as ordinary zero-dollar API usage:** loses the subscription quota dimension.

The lifecycle seam in [llm.py](/home/taygetea/code/regulatedconversation/motif/src/motif/llm.py:60) should remain authoritative. Concretely, add attachments/diagnostics to `CallCompleted` and `CallFailed`; let [record.py](/home/taygetea/code/regulatedconversation/motif/src/motif/record.py:65) project them.

## 3. V3 research shape

Yes, with two corrections:

```text
decompose
  → delegated structured investigations
  → deterministic evidence-integrity checks
  → targeted repair of failed evidence
  → delegated critique
  → synthesis from verified evidence packs
  → final citation-integrity check
```

The repair step matters. Otherwise invalid citations remain in briefs and critics can still reason from them.

### Evidence must be structured at research time

Transcript retention is necessary for forensics but insufficient for claim-level verification. Post-hoc transcript parsing has four problems:

- source-to-claim association may never have been stated;
- the final prose may combine several sources;
- parsing introduces another nondeterministic model step;
- tool traces may contain URLs but not the exact evidence relied upon.

Make the investigation result something like:

```json
{
  "brief": "...",
  "claims": [
    {
      "id": "collapse.wordsworth.threshold",
      "claim": "At Earth-like flux the modeled threshold is about 0.14 bar.",
      "source_url": "https://...",
      "source_title": "...",
      "locator": "abstract / figure 7 / page 12",
      "evidence_excerpt": "...",
      "support_kind": "direct_result",
      "accessed_at": "..."
    }
  ],
  "uncertainties": [...]
}
```

The transcript attachment then answers “how did the harness get here?” The structured claims answer “what evidence may downstream code rely on?”

### Do not overname deterministic verification

Code can deterministically check:

- URL resolution;
- DOI/arXiv/title/author/year identity;
- article numbers;
- exact excerpt occurrence;
- content hashes;
- arithmetic and unit conversions;
- whether a quoted number appears near the cited locator;
- duplicate or contradictory bibliographic records.

Code cannot generally determine that a source semantically entails an arbitrary scientific claim. Call the stage **deterministic evidence-integrity verification**, not deterministic truth or citation-support verification.

Semantic support still belongs to a critic/model/human—but the critic should receive claim IDs and verifier statuses, not only prose:

```json
{
  "claim_id": "...",
  "verifier_status": "source_resolved_quote_matched",
  "critique": "...",
  "counterevidence": [...]
}
```

The synthesizer should be instructed to cite claim IDs. Rendering can resolve those IDs into the verified URLs. This prevents synthesis from casually inventing a fresh author-year citation.

## 4. What breaks first

### First: the claimed read-only effect radius

The installed Codex CLI is `0.144.3`. It exposes `--sandbox read-only`, `--ephemeral`, `--ignore-user-config`, JSONL, and output schemas—but it does not expose a simple `--tools web_search,web_fetch,read` allowlist comparable to Claude’s `--tools`.

Furthermore, this machine’s Codex has apps, browser/computer use, MCP, hooks, plugins, and project configuration surfaces. A clean harness must demonstrate which remain enabled under the proposed flags. Until it does, `effect_ceiling=EXTERNAL_READ` is unproven.

The first implementation artifact should therefore be a capability/effect canary, not motif integration:

1. Run from a controlled empty cwd.
2. Use fresh/ephemeral execution.
3. Suppress user/project customizations as far as the CLI supports.
4. Request one web search.
5. Capture JSONL.
6. Assert the actual tool set used.
7. Assert no workspace mutation.
8. Assert no configured app/MCP/hook/plugin could write externally.
9. Record CLI/config fingerprint.

If Codex cannot expose or enforce that tool allowlist, the honest effect declaration is `EXTERNAL_UNKNOWN`, and by your own rule it cannot collapse to a read-only call node.

### Second: controller state versus model sandbox

My minimal test:

```text
codex exec --json --ephemeral --ignore-user-config --sandbox read-only ...
```

failed before inference because the in-process app-server client attempted a write in a location this consultation sandbox made read-only.

This exposes an important distinction:

- the **controller** needs writable runtime/auth/cache/temp state;
- the **model’s tools** must remain read-only.

Do not implement this by making all of `CODEX_HOME` writable to the model. The harness needs a private controller-owned runtime area that the delegated model cannot access as a writable tool path.

Also pass prompts through stdin with `create_subprocess_exec`, never a shell or interpolated argv. For Codex, use `PROMPT="-"`; otherwise piped stdin can be appended as a second `<stdin>` block.

### Third: quota and fan concurrency

Six parallel `codex exec` processes are not equivalent to six cheap HTTP calls. Each brings a large harness system context and internal turns. Codex’s transcript records subscription-window percentage and reset time. A fan can exhaust or throttle the shared window mid-run.

`max_concurrency` therefore belongs to the endpoint/profile, not each pipeline. Before starting a call, an admission controller should inspect the latest quota snapshot. Automatic fallback to paid API/Exa must be profile policy, never silent transport behavior.

### Fourth: clean mode may remove the capability you wanted

The bulk lane assumes “built-in web search.” But the exact clean invocation needed to prove read-only effects may disable the customization or feature that supplies web search. That needs an empirical integration test. Capability declarations cannot be aspirational.

### Fifth: CLI JSON schema drift

Treat JSONL as an adapter protocol, not a stable motif protocol:

- pin/test supported CLI versions;
- retain unknown events;
- parse cumulative token counts without summing every snapshot;
- require a recognizable terminal event plus valid result;
- attach raw stdout/stderr on every failure.

### Sixth: prompt semantics

`Msg` has typed system/user/tool segments. `codex exec` accepts an initial prompt inside Codex’s own system environment. This is not equivalent to the Anthropic API rendering.

The adapter must record both:

- original `Msg`;
- exact flattened prompt sent to the CLI.

Use explicit tagged sections and delimit untrusted source material. Do not pretend cross-profile system-prompt semantics are identical.

## Tomorrow-morning implementation order

1. Add `requires=` preflight to `complete` and `extract`.
2. Add `DelegatedEndpoint` as a distinct endpoint union member.
3. Refactor transport dispatch to return a common internal result containing output, usage, stop reason, attachments, and diagnostics.
4. Extend lifecycle events and `Node` with typed attachment references.
5. Implement the Codex JSONL adapter with ephemeral process handling, deadline, process-group kill, schema validation, and quota parsing.
6. Build the effect/capability canary. Do not mark the endpoint `EXTERNAL_READ` until it passes.
7. Implement `INVESTIGATION_SCHEMA` and evidence-integrity checks.
8. Refactor deep research to `extract(... requires={"web_read"})`.
9. Leave `flow.agent` routing untouched.

One final policy caveat: the fresh Codex manual fetch was blocked by this environment’s DNS restrictions, and official-domain search did not establish the broader “bulk programmatic subscription use is permitted” claim. `codex exec` is plainly a non-interactive automation surface, but that does not by itself settle subscription-at-scale policy. Keep that as deployment policy/configuration, not a motif invariant.