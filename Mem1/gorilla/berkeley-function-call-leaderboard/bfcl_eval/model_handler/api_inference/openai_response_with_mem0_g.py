# bfcl_eval/model_handler/api_inference/openai_response_with_memory.py
import json
import os
import time
import asyncio

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from openai.types.responses import Response
from typing import List, Any, Optional

from amem.memory_system import AgenticMemorySystem


class OpenAIResponsesHandlerWithMem0G(BaseHandler):
    MEM_TAG = "PRIVATE_MEMORY:"

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.model_name = model_name
        self.client = OpenAI(**self._build_client_kwargs())

        faiss_root = "/tmp/faiss_memories"
        kuzu_root = "/tmp/kuzu_graphs"
        os.makedirs(faiss_root, exist_ok=True)
        os.makedirs(kuzu_root, exist_ok=True)
        store_time = f"store_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        # Initialize the memory system 🚀
        self.config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.01,
                        "max_tokens": 2000,
                    }
                },
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "collection_name": "test",
                        "path": os.path.join(faiss_root, store_time),
                        "distance_strategy": "euclidean"
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                },
                "memory": {
                    "auto_embed": True,
                    "summarization": True,
                },
                "graph_store": {
                    "provider": "kuzu",
                    "config": {
                        "db": os.path.join(kuzu_root, store_time),
                    },
                }
            }

        self.memory_system = Memory.from_config(self.config)

    
    def _build_client_kwargs(self):
        kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if headers_env := os.getenv("OPENAI_DEFAULT_HEADERS"):
            kwargs["default_headers"] = json.loads(headers_env)
        return kwargs

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        for prompt in prompts:
            if prompt.get("role") == "system":
                prompt["role"] = "developer"
        return prompts

    def _reset_if_new_case(self, test_id: str):
        if test_id != self._cur_test_id:
            # push the lateset turn context to slots
            snapshot_events = _drain_snapshot(event_buffer=self._event_buffer)
            self.memory_system.add(snapshot_events, user_id="agent", infer=False)

    def _strip_old_memory_block(self, message: list[dict]) -> list[dict]:
        out = []
        for m in message:
            if m.get("role") == "developer" and isinstance(m.get("content"), str) and m["content"].startswith(self.MEM_TAG):
                continue
            out.append(m)
        return out

    def _inject_memory(self, query_text: str, message: list[dict]):

        returns = self.memory_system.search(query_text, limit=3, user_id="agent")
        if isinstance(returns, list):
            relevant_memories = returns
        else:
            relevant_memories = returns['results']
        print("1st relevant_memories:", relevant_memories[0]['memory'] if len(relevant_memories) > 0 else "None")
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)

        mem_msg = {
            "role": "developer",
            "content": (
                f"{self.MEM_TAG}\n"
                "Below is the agent's private memory from earlier turns in THIS test case.\n"
                "Use it only if helpful; do not mention it explicitly.\n\n"
                "Here are some relevant memories:\n"
                f"{memories_str}\n"
            ),
        }
        return [mem_msg] + message

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        message = self._strip_old_memory_block(message)
        message = self._normalize_messages(message)
        query_text = self._extract_latest_user_query_text(message)
        message = self._inject_memory(query_text=query_text, message=message)
        inference_data["message"] = message

        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "input": message,
            "model": self.model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }
        if ("o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name):
            del kwargs["temperature"]
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        self._reset_if_new_case(test_entry["id"])

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(test_entry["question"][round_idx])

        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools
        return inference_data

    def _parse_query_response_FC(self, api_response: Response) -> dict:
        model_responses = []
        tool_call_ids = []
        for func_call in api_response.output:
            if func_call.type == "function_call":
                model_responses.append({func_call.name: func_call.arguments})
                tool_call_ids.append(func_call.call_id)
        if not model_responses:
            model_responses = api_response.output_text

        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response.output,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        for m in first_turn_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        for m in user_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].extend(model_response_data["model_responses_message_for_chat_history"])
        memory = str(model_response_data["model_responses"])[:2000]
        _push_event(self._event_buffer, "ASSISTANT", memory)
        return inference_data

    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {"type": "function_call_output", "call_id": tool_call_id, "output": execution_result}
            inference_data["message"].append(tool_message)
            _push_event(self._event_buffer, "TOOL_RESULT", execution_result[:2000])
        return inference_data

    def _query_prompting(self, inference_data: dict):
        msg = inference_data["message"]
        msg = self._normalize_messages(msg)
        msg = self._strip_old_memory_block(msg)
        query_text = self._extract_latest_user_query_text(msg)
        print(f"[Debug] Query Text for Memory Injection: {query_text}")
        msg = self._inject_memory(query_text=query_text, message=msg)
        inference_data["message"] = msg

        inference_data["inference_input_log"] = {"message": repr(msg)}

        kwargs = {
            "input": msg,
            "model": self.model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }
        if ("o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name):
            del kwargs["temperature"]
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        self._reset_if_new_case(test_entry["id"])

        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(test_entry["question"][round_idx])

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Response) -> dict:
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": api_response.output_text,
            "model_responses_message_for_chat_history": api_response.output,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        for m in first_turn_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        for m in user_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].extend(model_response_data["model_responses_message_for_chat_history"])
        memory = str(model_response_data["model_responses"])[:2000]
        _push_event(self._event_buffer, "ASSISTANT", memory)
        return inference_data

    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        formatted_results_message = format_execution_results_prompting(inference_data, execution_results, model_response_data)
        inference_data["message"].append({"role": "user", "content": formatted_results_message})
        _push_event(self._event_buffer, "TOOL_RESULT", formatted_results_message[:2000])
        return inference_data

    from typing import Any

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

    def _content_to_text(self, content) -> str:
        # dict messages: content is usually str
        if isinstance(content, str):
            return content

        # Responses output message: content is often a list of parts with .text
        if isinstance(content, list):
            parts = []
            for p in content:
                t = getattr(p, "text", None)
                if isinstance(t, str):
                    parts.append(t)
            return "".join(parts)

        # fallback
        if content is None:
            return ""
        return str(content)


    def _normalize_one_message(self, m) -> dict:
        # Already a dict message
        if isinstance(m, dict):
            # ensure content is str (some code puts list parts here too)
            return {
                "role": m.get("role"),
                "content": self._content_to_text(m.get("content")),
            }

        # Responses API output object (pydantic): ResponseOutputMessage
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        return {
            "role": role,
            "content": self._content_to_text(content),
        }


    def _normalize_messages(self, msg):
        # msg should be list of messages
        if isinstance(msg, list):
            return [self._normalize_one_message(m) for m in msg]

        # Sometimes msg might be a single message
        return [self._normalize_one_message(msg)]

    def _extract_latest_user_query_text(self, message: list[dict]) -> str:
        """Get the latest user turn text as query_text for memory retrieval."""
        for m in reversed(message):
            if m.get("role") != "user":
                continue
            return self._flatten_user_content(m.get("content", ""))
        return ""