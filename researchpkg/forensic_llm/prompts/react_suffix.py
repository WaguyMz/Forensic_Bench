REACT_SUFFIX = """

---

## Action Format (text fallback mode)

Use EXACTLY this format when calling tools:

```
Thought: <your reasoning>
Action: <tool_name>
Action Input: <valid JSON object>
```

To finish:

```
Thought: I have gathered sufficient evidence.
Action: finish_investigation
Action Input: {"suspicion_list": [...], "narrative": "..."}
```

Do NOT emit any text between `Action:` and `Action Input:`.  Wait for the Observation.
"""
