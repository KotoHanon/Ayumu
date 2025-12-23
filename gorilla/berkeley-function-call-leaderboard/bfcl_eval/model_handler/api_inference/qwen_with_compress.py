import os
import json
import time
import asyncio
import logging
from typing import Any, List, Dict, Optional

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from openai import OpenAI
from overrides import override
from qwen_agent.llm import get_chat_model
from datetime import datetime
import time

from memory.memory_system.utils import (
    _push_event,
    _drain_snapshot,
    _safe_dump_str,
    _multi_thread_run,
)


class QwenAgentThinkHandlerWithCompress(OpenAICompletionsHandler):

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name,  # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8014/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': True
                }
            },
            "thought_in_content": True,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'max_input_tokens': 58000,
            'timeout': 1000,
            'max_tokens': 4096
        }
    })

    def _build_client_kwargs(self):
        kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if headers_env := os.getenv("OPENAI_DEFAULT_HEADERS"):
            kwargs["default_headers"] = json.loads(headers_env)
        return kwargs

    #### FC methods ####
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        
        inference_data["message"] = message

        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        start_time = time.time()
        if len(tools) > 0:
            responses = None
            for resp in self.llm.quick_chat_oai(message, tools):
                responses = resp 
                
        else:
            responses = None
            for resp in self.llm.quick_chat_oai(message):
                responses = resp
        end_time = time.time()
        
        return responses, end_time-start_time

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {"role": "tool", "tool_call_id": tool_call_id, "content": execution_result}
            inference_data["message"].append(tool_message)
        return inference_data

    
    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call['function']['name']: func_call['function']['arguments']}
                for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
            tool_call_ids = [
                func_call['function']['name'] for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
        except:
            model_responses = api_response["choices"][0]["message"]["content"]
            tool_call_ids = []
        
        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response["choices"][0]["message"],
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.get("usage", {}).get("prompt_tokens", 0),
            "output_token": api_response.get("usage", {}).get("completion_tokens", 0),
        }
        return response_data
        

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        
        if isinstance(model_response_data["model_responses_message_for_chat_history"], list):
            inference_data["message"]+=model_response_data["model_responses_message_for_chat_history"]
        else:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"]
            )

        if len(model_response_data.get("tool_call_ids", [])) == 0:
            # No tool calls means that the end of turn
            self._compress_history_message(inference_data)

        return inference_data

    def _flatten_user_content(self, content: Any) -> str:
        """Flatten user message content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()

        # Some libs use list[{"type": "...", "text": "..."}]
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    # common keys: {"type":"text","text":...} / {"type":"input_text","text":...}
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    # some variants might use {"type":"text","content":...}
                    elif "content" in item and isinstance(item["content"], str):
                        parts.append(item["content"])
            return "\n".join(p.strip() for p in parts if p and p.strip()).strip()

        # fallback (dict or others)
        if isinstance(content, dict):
            # rare: {"text": "..."} or {"content": "..."}
            for k in ("text", "content"):
                v = content.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""


    def _compress_history_message(self, inference_data: dict) -> dict:
        message = inference_data["message"]
        prompt = f"""You are a history summarizer for a task-oriented tool-using agent (BFCL setting).

Input: a full conversation history containing messages from roles: system, user, assistant, and tool (tool results).
Output: ONE single plain-English string (no JSON, no Markdown) that will be inserted as an assistant message and used as the compressed history for the next turn.

Hard rules:
- Do NOT invent or assume missing information. If something is unknown, explicitly say "UNKNOWN".
- Preserve anything needed to correctly continue the task and to form valid tool calls.
- If any future tool call would require required arguments that are not explicitly provided in history, you MUST state which arguments are missing and that the agent should ask the user for them (do not guess).
- Keep only decision-critical content; drop greetings, chit-chat, and repeated phrasing.
- Tool results can be long: keep only reusable key fields (IDs, numbers, paths, final verdicts) and any error messages/causes.
- Keep it concise: target ~80–180 words.

Return the summary in EXACTLY this template (still as a single string):

COMPRESSED_HISTORY:
[GOAL] <the user’s current objective in one sentence>
[STATE] Done: <what has been completed> | Now: <current situation/progress>
[FACTS] <key facts/variables/choices confirmed so far; include timestamps if present>
[TOOLS] <most recent tool calls and key outcomes/errors; include IDs/paths/numbers>
[CONSTRAINTS] <non-negotiable rules from system/user/tool schemas; “no guessing”, formats, limits>
[MISSING] <required info not present; list exact missing args/questions; or “None”>
[NEXT] <the next best action; include specific clarification questions if needed>

Here is the conversation history:
{_safe_dump_str(message)}
"""

        compression_response = None
        for res in self.llm.quick_chat_oai([{"role": "user", "content": prompt}]):
            compression_response = res

        if compression_response is None:
            raise RuntimeError("No response from quick_chat_oai")

        summary = compression_response["choices"][0]["message"].get("content", "")
        summary = summary.strip()

        inference_data["message"] = [
            {"role": "user", "content": "Here is the compressed conversation history."},
            {"role": "assistant", "content": summary},
            ]
        return inference_data