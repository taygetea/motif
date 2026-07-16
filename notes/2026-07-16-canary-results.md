# Codex capability canary — results (2026-07-16, codex-cli 0.144.3)

Gate for the delegated-endpoint work (see 2026-07-16-delegated-endpoints-sol.md).
All probes from an empty cwd. Verdict: **lane viable; `mcp_servers={}` is
mandatory, not optional.**

## The hermetic invocation

```
codex exec --ephemeral --ignore-user-config --skip-git-repo-check \
    -s read-only --json -c 'mcp_servers={}' -c tools.web_search=true \
    "<prompt>" < /dev/null
```

1. **stdin must be closed** (or the prompt passed as `-` via stdin): with a
   non-TTY stdin left open, `codex exec` blocks on "Reading additional
   input from stdin..." until killed. Sol predicted this.
2. **Web search survives clean mode.** `-c tools.web_search=true` works
   under `--ignore-user-config`; the search appears as a `web_search`
   item in the JSONL with the query visible. (Refutes breaks-first #4.)
3. **CRITICAL — the default "clean" invocation is EXTERNAL_WRITE.**
   `--ignore-user-config` does NOT strip ChatGPT app connectors: the tool
   list included `mcp__codex_apps__github_add_comment_to_issue`,
   `add_issue_assignees`, `add_reaction_to_pr`, etc. — live remote writes
   on the account's connected GitHub. The `read-only` sandbox governs
   LOCAL shell only. `-c 'mcp_servers={}'` removes the entire app
   surface. (Confirms breaks-first #1, worse than predicted.)
4. **Residual tool set after the strip** (enumerated, schemas quoted by
   the model itself): exec/apply_patch (local, sandbox-governed — a
   write attempt produced no file), `web__run` ("tool for accessing the
   internet", read-only per its schema), `image_gen__imagegen` (content
   generation, no external state mutation), `request_plugin_install`
   (suggest-only), `collaboration.*` (spawns more codex agents — quota
   multiplication, same ceiling). Honest ceiling: **EXTERNAL_READ +
   content generation**; the attestation must carry this enumerated list.
5. **Full-disk read under `-s read-only`** — it read `~/.profile` on
   request. Fine for the single-experimenter trust model; must appear in
   the attestation (`fs_read: full_disk`); disqualifying for any
   multi-tenant future without a stricter sandbox.
6. **Usage is in the JSONL**: `turn.completed` carries
   `{input_tokens, cached_input_tokens, output_tokens,
   reasoning_output_tokens}`. No quota-window snapshot observed in these
   event types — Sol's rate-limit-snapshot claim may refer to session
   files; open item for the adapter's usage_policy.
7. **Concurrency: genuinely parallel.** Three simultaneous execs: 4.8s /
   5.3s / 7.3s individually, 7.3s total wall (max, not sum). No
   serialization or rate-limit errors at n=3 on this account. Larger
   fan widths unmeasured.

## Event vocabulary observed

`thread.started`, `turn.started`, `item.started`,
`item.completed` (item.type ∈ web_search, command_execution,
agent_message), `turn.completed` (usage). Treat as adapter protocol;
pin the CLI version in the invocation fingerprint.

Probes archived in the session scratchpad (probe1/2/2b/2c, conc1-3).
