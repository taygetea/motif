# Backlog: outside-scope / deferred observations

- `display.py` has a private `_Node` dataclass distinct from `graph.Node` — same name,
  different concept, easily confused. Consider renaming to `_TreeNode`.
- README line 167 already names the Runtime-object seam for observer isolation; if B1
  (global graph state) gets fixed, consider doing the observer lists (`llm._observers`,
  `flow._observers`) in the same pass — same design smell, explicitly acknowledged in README.
- Static-check the 7 examples other than deep_research_v2.py against current API
  (agent_compose, blackboard, dialectic, prism, temporal_analysis, tree_decomposition,
  tui_demo, tui_full) — only deep_research_v2 was verified in this pass.
- CostTracker `_PRICING` table currency should be re-checked against current model IDs
  when touching C1.
- Decide: is `backend="openai"` on llm verbs (A5) a feature to finish or a kwarg to remove?
  Blocks planning for that item.
