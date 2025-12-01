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

Assess three dimensions for whether to store a memory. Your goal is to keep any information that can help answer similar questions in the future, not only groundbreaking discoveries.

1. Question relevance – Does this information help answer the current question, explain a reasoning step, or connect two entities/events that the question cares about?
   - Examples: "X is the capital of Y", "Person A is the parent of Person B", "Event C happened before Event D".

2. Reusable fact or pattern – Could a similar question in the future reuse this fact, relation, or short reasoning chain?
   - This includes:
     - Entity attributes (dates, locations, roles, definitions).
     - Relations between entities (A part_of B, A causes B, A located_in B).
     - Short multi-hop links that serve as a bridge between pieces of evidence.

3. Reliability from context – Is this information explicitly stated or clearly implied by the given context, rather than purely guessed?
   - Prefer facts supported by sentences in the context over speculative statements.

Decision rule:
- Store as memory if it is clearly relevant to the question (dimension 1) AND it satisfies at least one of dimensions 2 or 3.
- Do NOT store:
  - Pure logging / formatting / boilerplate ("Now I will think step by step", "Searching Wikipedia...").
  - Extremely local details that do not help answer questions (token-level noise, partial URLs, random IDs).

Return `yes` only when at least two dimensions are clearly satisfied or the slot closes a critical loop (e.g., root-causing a failure, finishing a checklist item). Otherwise return `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Map this WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic: enduring insights, generalized conclusions, reusable heuristics.
- episodic: Situation → Action → Result traces with metrics, timestamps, or narrative context.
- procedural: MUST provide "when to use", and OPTIONAL reproducible steps, commands, pipelines, or interface contracts that express "how to".

Tie-breaking rules:
- Prefer episodic if a chronological action/result trail exists, even if insights appear.
- Prefer procedural when explicit steps/tools/commands are primary.
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
Convert the QA Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

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
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤130 words and make it self-contained.
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
<working-slots>
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
</working-slots>
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
<working-slots>
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
</working-slots>
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

#
'''TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval.

Expectations:
- `summary` (≤80 words) expresses the enduring conclusion or heuristic.
- `detail` should elaborate supporting evidence, metrics, or caveats. Use "\\n" to separate logically distinct statements.
- `tags` mixes domain terms and method/process hints.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<semantic-record>
{{
    "summary": "semantic insight summary",
    "detail": "expanded reasoning and context",
    "tags": ["keyword1","keyword2"]
}}
</semantic-record>
"""
)'''
#

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
