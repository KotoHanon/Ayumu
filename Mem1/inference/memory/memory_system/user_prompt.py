from textwrap import dedent

WORKING_SLOT_EXPERIEMENT_FILTER_USER_PROMPT = dedent("""
You guard ResearchAgent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Assess four dimensions:
1. Novelty – is this meaningfully new compared to typical research agent discoveries?
2. Utility – can future tasks reuse the insight, metric, procedure, or decision?
3. Stability – will the information stay valid for multiple iterations (i.e., not a transient log)?
4. Evidence – do attachments, metrics, or tags provide concrete support?

Return `yes` only when at least two dimensions are clearly satisfied or the slot closes a critical loop (e.g., root-causing a failure, finishing a checklist item). Otherwise return `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_QA_FILTER_USER_PROMPT = dedent("""
You guard QA-Agent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Your goal is to keep **reusable, high-level memories** that can help answer future questions in the same domain (e.g., HotpotQA-style multi-hop QA), not just low-level logs.

Evaluate the slot along three dimensions:

1. Reusable knowledge or pattern – Does this slot contain a fact, relation, strategy, or failure pattern that could be helpful for similar questions in the future?
   - Includes:
     - Entity attributes (dates, locations, roles, definitions).
     - Relations between entities (A part_of B, A located_in B, A spouse_of B, A causes B).
     - Short multi-hop bridges that connect entities or events.
     - Generalizable QA strategies or common failure modes.

2. Abstraction level – Is the content more than raw log text?
   - Prefer:
     - Summaries of reasoning steps, heuristics, or patterns.
     - Condensed statements that compress multiple observations.
   - Avoid:
     - Pure boilerplate (“now I will think step by step”, “searching Wikipedia…”).
     - Very low-level noise (token indices, random IDs, partial URLs) with no clear use.

3. Reliability and stability – Is the information well-supported by the context and relatively stable over time?
   - Prefer:
     - Facts or patterns that are explicitly stated or clearly implied.
   - Avoid:
     - Pure guesses, opinions, or very ephemeral states.

Decision rule:
- Default is slightly conservative, but you SHOULD store any slot that carries at least one **non-trivial, reusable** fact, relation, or strategy.
- Return `yes` if:
  - The slot contains **some reusable knowledge or pattern** (dimension 1), AND
  - It is not dominated by pure logging/noise, AND
  - It is at least moderately reliable (dimension 3) OR shows some abstraction (dimension 2).
- Return `no` if the slot is:
  - Mostly boilerplate/logging,
  - Extremely local and unlikely to be reused,
  - Or dominated by noise/IDs without a clear semantic core.

When you are uncertain BUT the slot contains at least one meaningful, reusable fact or strategy, prefer answering `yes` rather than `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_FC_FILTER_USER_PROMPT = dedent("""
You guard a Function-Calling agent's long-term memory entrance. Decide if this candidate memory deserves promotion into FAISS storage.

Your goal is to keep **reusable, high-level function-calling knowledge** that improves future tool-use (schema grounding, argument filling, error recovery, tool routing), not low-level logs.

Evaluate the candidate along three dimensions:

1. Reusable tool-use knowledge or pattern – Does it contain a rule, constraint, mapping, or recovery strategy that can help future function calls?
   - Includes:
     - Tool schema constraints (required fields, enums, type constraints, allowed ranges).
     - Argument-filling strategies (how to infer/ask for missing required args; defaults; disambiguation).
     - Tool selection/routing heuristics (when to use tool A vs B; order of tools).
     - Common failure modes and fixes (400 invalid_request_error, missing tool output, bad JSON, rate limits, retries).
     - Output-parsing templates (how to robustly extract JSON; handle code fences; validate before loads).
     - Multi-turn FC protocol patterns (call_id pairing, tool_call -> tool_output -> next model call).

2. Abstraction level – Is it more than a one-off trace?
   - Prefer:
     - General rules, checklists, invariants, and debugging playbooks.
     - Minimal, canonical examples illustrating the rule.
   - Avoid:
     - Raw transcripts, stack traces, UUIDs, request IDs, or file paths **without** a generalized lesson.
     - Exact tool outputs that are not reusable.
     - “We tried X once” with no stable takeaway.

3. Reliability and stability – Is it correct and likely to remain useful?
   - Prefer:
     - Behaviors required by the FC protocol (e.g., tool_call must have matching tool_output).
     - Constraints directly grounded in tool specs or repeated observations.
   - Avoid:
     - Speculation about model internals.
     - Ephemeral environment quirks (temporary outages) unless generalized as a robust fallback.

Decision rule:
- Default is slightly conservative, but you SHOULD store any candidate that carries at least one **non-trivial, reusable** FC rule or strategy.
- Return `yes` if:
  - The candidate includes some reusable tool-use knowledge/pattern (dimension 1), AND
  - It is not dominated by pure logs/noise, AND
  - It is at least moderately reliable (dimension 3) OR shows clear abstraction (dimension 2).
- Return `no` if the candidate is:
  - Mostly a one-off trace/log with no generalized rule,
  - Extremely local to a single run/filepath/request-id,
  - Or dominated by noise/IDs without a semantic core.

When uncertain BUT the candidate contains at least one meaningful, reusable FC constraint or debugging strategy, prefer answering `yes` rather than `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")


WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Map this WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic: enduring insights, generalized conclusions, reusable heuristics.
- episodic: Situation → Action → Result traces with metrics, timestamps, or narrative context.
- procedural: step-by-step checklists, reusable commands, or skill blueprints.

Tie-breaking rules:
- Prefer episodic if a chronological action/result trail exists, even if insights appear.
- Otherwise output semantic.

Return only one of: "semantic", "episodic", "procedural".

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_COMPRESS_USER_PROMPT = dedent("""
Merge the provided WorkingSlots into ONE distilled WorkingSlot suitable for the short-term queue.

Requirements:
- Remove duplicate facts while keeping supporting metrics or attachments that future agents might need.
- Surface causal links (Situation → Action → Result) whenever present.
- Normalize tags to 1–4 lowercase tokens.
- Keep summary ≤150 words; emphasize reusable, stable insights spanning research, execution, and follow-up actions.
- If attachments include command snippets, metrics, or notes, fold only the most representative subset into the compressed slot.

Input WorkingSlots (JSON):

<slots>
{slots_block}
</slots>

Output format (STRICTLY JSON):
<compressed-slot>
{{
    "stage": "compressed",
    "topic": "concise topic slug",
    "summary": "≤150 words describing the merged knowledge",
    "attachments": {{
        "notes": {{"items": ["bullet 1","bullet 2"]}},
        "metrics": {{"name": value}},
        "procedures": {{"steps": ["step1","step2"]}},
        "artifacts": {{"paths": ["..."]}}
    }},
    "tags": ["tag1","tag2"]
}}
</compressed-slot>
""")

ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT = dedent("""
You aggregate episodic traces into a single semantic memory entry. Capture the durable lesson that explains why the cluster exists.

Instructions:
- Highlight causal mechanisms, success/failure thresholds, and metrics that repeatedly appeared.
- Mention representative stages (e.g., experiment_execute) only if they add meaning.
- Provide tags that cover both domain concepts and process cues (e.g., ["vision","fog","stability"]).
- Return STRICT JSON containing `summary`, `detail`, `tags`.

Episodic cluster notes:
{episodic_notes}
""")

TRANSFER_SLOT_TO_TEXT_PROMPT = dedent("""
Convert the WorkingSlot JSON into a concise human-readable paragraph (no tags, no JSON). This summary feeds chat surfaces, not FAISS.

Guidance:
- Mention stage, topic, and the core outcome or decision.
- Cite standout metrics or attachments inline (e.g., "accuracy climbed to 0.73").
- Describe actionable next steps only if explicitly recorded.
- Limit to 2–4 sentences; avoid bulleting or markdown.

Input WorkingSlot (JSON):

{dump_slot_json}
""")

TRANSFER_QA_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT = dedent("""
Convert the QA Agent workflow context into at most {max_slots} WorkingSlot and at least 1 WorkingSlot entries ready for filtering/routing.

Goal:
- Only create slots that are worth storing for long-term reuse.
- A slot SHOULD be created if and only if it is:
  (A) Semantic evidence: stable, factual content (e.g., retrieved passages, environment observations) that can later support answering other questions.
  OR
  (B) Episodic experience: reusable process knowledge, strategy, or failure pattern about how to solve HotpotQA-style multi-hop questions.

Ignore:
- Ephemeral chatter, low-level execution logs, or transient chain-of-thought that does not generalize.
- Speculation not grounded in retrieved content or clear experience.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (one decision, discovery, bottleneck, or command).
2. `stage` MUST be one of: question_understanding, information_retrieval, answer_generation, answer_validation, meta.
   - Use `information_retrieval` / `answer_validation` often for semantic evidence.
   - Use `question_understanding` / `answer_generation` / `meta` often for episodic experience.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤80 words and make it self-contained.
4. `topic` is a 3–6 word slug referencing the problem space (lowercase, space-separated), e.g. "bridge entity retrieval", "evidence aggregation failure".
5. `attachments` is optional but, when present, group similar info under keys such as:
    - "notes": {{"items": []}}        # short bullet-like notes or paraphrased facts
    - "references": {{"links": []}}   # source titles, IDs, or URLs (e.g. Wikipedia pages)
    - "issues": {{"list": []}}       # open problems, errors, or caveats
    - "actions": {{"list": []}}       # next steps or commands
6. `tags` is a list of lowercase keywords (≤5 items) mixing:
    - domain hints: "hotpotqa","wikipedia","multi-hop","bridge-entity"
    - workflow hints: "semantic-evidence","episodic-experience","retrieval","planning","verification","failure"
   Use "semantic-evidence" for factual slots and "episodic-experience" for process/strategy slots.

Output STRICTLY as JSON within the tags below (no extra commentary):
{{
    "slots": [
        {{
            "stage": "information_retrieval",
            "topic": "bridge entity localization",
            "summary": "Situation/Action/Result narrative focusing on a single reusable semantic evidence or episodic pattern.",
            "attachments": {{
                "notes": {{"items": ["short fact or note 1", "short fact or note 2"]}},
                "references": {{"links": ["wikipedia:Albert_Einstein"]}},
                "issues": {{"list": []}},
                "actions": {{"list": ["optional follow-up"]}}
            }},
            "tags": ["hotpotqa","semantic-evidence","retrieval"]
        }}
    ]
}}
""")

TRANSFER_FC_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT = dedent("""
Convert the BFCL tool-using agent trajectory into at most {max_slots} WorkingSlot entries ready for filtering/routing.

BFCL characteristics to account for:
- Multiple candidate tools may be present; only some are relevant.
- Required parameters may be missing; the safe behavior is to ask for clarification (not guess).
- Multi-step: tool outputs in earlier steps often constrain later arguments.
- Some system messages explicitly forbid assumptions; preserve such constraints as evidence.
- Tool descriptions / schemas can be noisy; store disambiguation rules as procedural experience.

PRIMARY GOAL (IMPORTANT):
Produce reusable, tool-agnostic knowledge. Prefer turning trajectory traces into:
1) PROCEDURAL playbooks (checklists / steps you can reuse for new tools and new tasks),
2) SEMANTIC invariants (constraints, schema rules, arg mappings grounded in evidence),
and ONLY THEN episodic lessons (one-off “what happened” stories).

Slot composition priority (VERY IMPORTANT):
- Prefer PROCEDURAL and SEMANTIC slots over EPISODIC.
- You MUST produce at least:
  - 2 procedural-experience slots IF any tool call occurred in the snapshot,
  - 1 semantic-evidence slot IF any explicit constraint/schema/output exists.
- Create episodic-experience slots ONLY if they contain a failure mode or tactic that cannot be expressed as a general checklist/playbook.
- If a takeaway can be written as a tool-agnostic rule/checklist/playbook, DO NOT write it as episodic.

Never store:
- Any gold labels or evaluator-only signals (if present).
- Raw chain-of-thought or verbose inner reasoning; keep summaries outcome-focused.
- Overly specific private strings that are not reusable (e.g., full addresses/IDs) unless they illustrate a general rule. Prefer anonymized patterns.

Context Snapshot (may include dialogue history, tool schemas, tool outputs):
<bfcl-context>
{snapshot}
</bfcl-context>

Authoring rules:
1. Each slot MUST capture exactly ONE reusable takeaway.
2. `stage` MUST be one of:
   - intent_constraints         # user intent + hard constraints (no sugar, metric units, etc.)
   - tool_selection             # selecting the right tool among distractors
   - argument_construction      # mapping text -> required args, filling enums, defaults policy
   - tool_execution             # calling tools, handling returned outputs
   - result_integration         # merging tool outputs into next-turn state/answer
   - error_handling             # retries, validation failures, unsupported protocol, etc.
   - meta                       # general insights, evaluation protocol concerns
3. `summary` MUST be ≤90 words, self-contained, and use the format depending on memory type:
   - Procedural (preferred): Goal → Preconditions → Steps → Checks (compact, imperative).
   - Semantic: Invariant/Constraint → Evidence → Implication.
   - Episodic (only if needed): Situation → Action → Result.
4. `topic` is a 3–7 word slug (lowercase, space-separated),
   e.g. "no-assumption parameter filling", "tool distractor disambiguation".
5. `attachments` is optional but, when present, use these keys when relevant:
   - "constraints": {{"items": []}}     # explicit do/don't rules (e.g., "ask for clarification")
   - "tool_schema": {{"items": []}}     # compact schema notes: required fields, enums, defaults policy
   - "arg_map": {{"items": []}}         # text-to-arg mapping patterns
   - "observations": {{"items": []}}    # key tool outputs / environment facts (paraphrased)
   - "failures": {{"items": []}}        # error symptoms and likely causes
   - "recovery": {{"steps": []}}        # REQUIRED for procedural slots: a reusable playbook (not only for failures)
   - "checks": {{"items": []}}          # validation checks before/after calling tools
6. For PROCEDURAL slots:
   - You MUST include "recovery": {{"steps": [...]}} as a general reusable SOP (3–7 steps).
   - Steps should be tool-agnostic when possible (e.g., "validate required fields", "confirm enums", "pair call_id").
7. For SEMANTIC slots:
   - Prefer filling "constraints"/"tool_schema"/"arg_map"/"observations" with compact bullets grounded in the snapshot.
8. For EPISODIC slots:
   - Keep them rare; must include a novel failure mode and a reusable fix not already covered by procedural playbooks.
9. `tags` is a list of lowercase keywords (≤6) mixing:
   - domain: "bfcl","function-calling","multi-turn","tool-use"
   - memory type: "semantic-evidence","episodic-experience","procedural-experience"
   - skill: "tool-selection","arg-filling","clarification","validation","error-recovery","distractor-tools"

Routing hints (implicit, do not add extra fields beyond schema):
- semantic: store stable constraints, schemas, outputs, invariant mappings.
- episodic: store rare strategies/failures that generalize but cannot be expressed as a generic SOP.
- procedural: store step-by-step playbooks (pre-checks, arg filling, calling, interpreting, retrying).

Output STRICTLY as JSON within the tags below (no extra commentary):
{{
  "slots": [
    {{
      "stage": "argument_construction",
      "topic": "no-assumption required-args protocol",
      "summary": "Goal: produce a valid tool call without guessing. Preconditions: required fields missing in user text. Steps: ask for the missing fields explicitly; confirm enums; restate inferred args; request confirmation if ambiguous. Checks: verify all required fields present; verify enum values allowed. Result: avoids invalid requests and respects no-assumption rules.",
      "attachments": {{
        "constraints": {{"items": ["do not guess missing required args; ask targeted clarification"]}},
        "recovery": {{"steps": ["identify required args from schema", "compare against user-provided info", "ask a single clarification listing missing args", "confirm enum/value constraints", "construct call with only validated fields", "re-validate before sending"]}},
        "checks": {{"items": ["all required fields present", "enum values allowed", "no conflicting constraints"]}}
      }},
      "tags": ["bfcl","function-calling","procedural-experience","arg-filling","clarification","validation"]
    }}
  ]
}}
""")


TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent("""
Convert the Experiment Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (decision, discovery, bottleneck, or command).
2. `stage` MUST be one of: pre_analysis, code_plan, code_implement, code_judge, experiment_execute, experiment_analysis, meta.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤130 words.
4. `topic` is a 3–6 word slug referencing the problem space.
5. `attachments` is optional but, when present, group similar info under keys such as
   - "notes": {{"items": []}}
   - "metrics": {{}}
   - "issues": {{"list": []}}
   - "actions": {{"list": []}}
6. `tags` is a list of lowercase keywords (≤5 items) mixing domain + workflow hints.
7. If the context lacks meaningful content, return `"slots": []` but keep the envelope.

Output STRICTLY as JSON within the tags below:
{{
    "slots": [
    {{
        "stage": "code_plan",
        "topic": "coverage planning",
        "summary": "Situation/Action/Result narrative...",
        "attachments": {{
            "notes": {{"items": ["detail 1", "detail 2"]}},
            "metrics": {{"acc": 0.92}},
            "issues": {{"list": []}},
            "actions": {{"list": ["follow-up 1"]}}
        }},
        "tags": ["plan","coverage"]
    }}
    ]
}}
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in HotpotQA-style multi-hop QA.

Expectations:
- The semantic record MUST capture **factual evidence** grounded in the WorkingSlot (e.g., retrieved passages, environment observations), not planning logic or agent-control flow.
- `summary` (≤80 words) is a compact, question-agnostic factual statement or tightly related fact cluster that can be reused as evidence (e.g., key relations, attributes, dates, locations).
- `detail` elaborates the supporting evidence: paraphrased or briefly quoted spans, source/page titles or IDs, and important caveats. Use "\\n" to separate logically distinct atomic facts or evidence items.
- Avoid speculation or heuristic advice; only include information that is directly supported by the WorkingSlot content.
- `tags` should mix entity names, domain hints, and relation/type hints (e.g., ["hotpotqa","wikipedia","albert-einstein","birthplace"]).

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags, with no extra text:
<semantic-record>
{{
    "summary": "semantic evidence summary",
    "detail": "expanded factual evidence and context",
    "tags": ["keyword1","keyword2"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing Situation → Action → Result.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word Situation → Action → Result overview",
    "detail": {{
        "situation": "Context and constraints",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "metrics": {{}},
        "artifacts": []
    }},
    "tags": ["keyword1","keyword2"]
}}
</episodic-record>
""")


TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable skill or checklist.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "short skill name",
    "description": "≤60 words explaining when/why to apply it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["keyword1","keyword2"]
}}
</procedural-record>
""")
