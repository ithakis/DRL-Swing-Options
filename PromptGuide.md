# GPT-5.2 Prompting Guide — Practical Reference (Personal Notes)

## 1. Introduction (What to expect)
- GPT-5.2 tends to be more disciplined, concise, and structure-forward.
- Still prompt-sensitive: explicit constraints (scope, verbosity, format) materially improve reliability. 

## 2. Key behavioral differences (What changes in practice)
- More deliberate scaffolding and plan-like structure by default.
- Generally lower verbosity; you must *specify* when you want more detail.
- Stronger instruction adherence; less drift when constraints are explicit.
- Tool-use can be more “eager” in interactive flows; you can steer for fewer calls.

---

## 3. Prompting patterns

### 3.1 Controlling verbosity and output shape
Use an explicit verbosity clamp with an output “shape”.

<output_verbosity_spec>
- Default: 3–6 sentences OR ≤5 bullets.
- Simple yes/no: ≤2 sentences.
- Complex multi-step: 
  - 1 short overview paragraph
  - then ≤5 bullets tagged:
    - What changed
    - Where
    - Risks
    - Next steps
    - Open questions
- Prefer compact bullets over long narrative.
- Do not rephrase my request unless necessary to change meaning.
</output_verbosity_spec>

### 3.2 Preventing scope drift (especially frontend/UX tasks)
<design_and_scope_constraints>
- Implement EXACTLY what I request; no extra features.
- Do not add UI elements, styling flourishes, animations, tokens, or “nice-to-haves” unless asked.
- If ambiguous, choose the simplest valid interpretation and proceed.
</design_and_scope_constraints>

### 3.3 Long-context and recall (“lost in the scroll” prevention)
<long_context_handling>
- If input is long (multi-doc / multi-PDF / long threads):
  - First: produce a brief outline of relevant sections.
  - Restate my constraints (jurisdiction, date range, product, etc.).
  - Anchor claims to specific sections (“In the Data Retention section…”).
  - If details matter (dates/thresholds/clauses), quote minimally or paraphrase precisely.
</long_context_handling>

### 3.4 Ambiguity & hallucination risk controls
<ambiguity_handling>
- If requirements are missing or contradictory: ask targeted questions.
- If fresh facts are needed but no tools are available: say so explicitly and propose next steps.
- Never guess IDs, URLs, metrics, policy text, or figures.
</ambiguity_handling>

---

## 4. Compaction (Extending Effective Context)
Use compaction when:
- Multi-step agent flows have many tool calls.
- Long conversations must be retained beyond context limits.
Guidance:
- Compact after milestones, not every turn.
- Treat compacted artifacts as continuation state, not human-inspectable notes.

(Implementation lives in the Responses API “compact” endpoint; keep this as an ops playbook item.)

---

## 5. Agentic steerability & user updates (status updates without noise)
<user_updates_spec>
- Provide brief updates (1–2 sentences) only when:
  - a new major phase starts, OR
  - you discover something that changes the plan.
- Avoid narrating routine tool calls.
- Each update must include at least one concrete outcome (“Found X”, “Confirmed Y”, “Updated Z”).
- Do not expand scope; optional work must be clearly labeled optional.
</user_updates_spec>

---

## 6. Tool-calling and parallelism
<tool_usage_rules>
- Prefer tools over assumptions when facts may be uncertain or user-specific.
- Parallelize independent reads/searches when possible.
- After any write/update tool call, restate:
  - What changed
  - Where (ID/path)
  - Validation performed
</tool_usage_rules>

---

## 7. Structured extraction (PDF / Office / tables)
<extraction_spec>
You will extract structured data into JSON.
- Follow the schema exactly (no extra fields).
- Required vs optional fields must be explicit.
- If a field is missing: set to null; do not guess.
- Re-scan the source once before returning to reduce omissions.
</extraction_spec>

Example schema:
{
  "party_name": "string",
  "jurisdiction": "string|null",
  "effective_date": "string|null",
  "termination_clause_summary": "string|null"
}

---

## 8. Prompt migration guide to GPT-5.2 (operational checklist)
1) Switch models first; don’t change the prompt yet.
2) Pin reasoning effort explicitly to preserve latency/cost profile.
3) Run evals as baseline.
4) If regressions:
   - tighten verbosity/format/schema constraints
   - tighten scope constraints
   - adjust reasoning effort one notch at a time
5) Re-run evals after each change.

---

## 9. Web search and research prompting
<web_search_rules>
- Prefer research over assumptions when facts may be uncertain.
- Resolve contradictions across sources.
- Follow second-order leads until marginal value drops.
- Provide citations for web-derived claims.
</web_search_rules>

---

## 10. Conclusion (how to use this in production)
- Treat prompts as versioned assets.
- Pin reasoning effort + output shape.
- Use evals to validate changes; change one variable at a time.
