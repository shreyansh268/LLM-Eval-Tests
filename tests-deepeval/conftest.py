"""
conftest.py — Shared fixtures for all DeepEval tests.

Key design decisions:
  - judge_llm uses format="json"  → Ollama forces JSON output at API level
  - chat_llm  uses plain text     → for generating actual_output in test cases
  - _extract_json()               → strips markdown fences / prose wrappers if
                                    the model still leaks non-JSON around the object
  - Session-scoped fixture        → model is created once, reused across all files
"""

import json
import re
from typing import Optional

import pytest
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from deepeval.models.base_model import DeepEvalBaseLLM

# ── Default connection settings (override via env if needed) ─────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    """
    DeepEval-compatible Ollama wrapper with robust JSON handling.

    Two internal LLM instances:
      judge_llm — format="json" so Ollama guarantees a JSON token stream.
                  Used by DeepEval metrics (AnswerRelevancy, Faithfulness …).
      chat_llm  — plain text.
                  Used when generating actual_output for test cases.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        # Judge: JSON mode forces the model to output valid JSON structure
        self.judge_llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            format="json",
        )
        # Chat: plain text for normal generation
        self.chat_llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
        )

    # ── DeepEvalBaseLLM required interface ────────────────────────────────────

    def load_model(self):
        return self.judge_llm

    def generate(self, prompt: str) -> str:
        """
        Called internally by every DeepEval metric to score/judge outputs.
        Uses JSON mode + extraction so small models don't break the scorer.
        """
        raw = self.judge_llm.invoke([HumanMessage(content=prompt)]).content
        return self._extract_json(raw)

    async def a_generate(self, prompt: str) -> str:
        raw = (
            await self.judge_llm.ainvoke([HumanMessage(content=prompt)])
        ).content
        return self._extract_json(raw)

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

    # ── Plain-text generation for test case actual_output ─────────────────────

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """
        Generate a natural-language response (NOT JSON-mode).
        Use this to create actual_output values inside test cases.
        """
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return self.chat_llm.invoke(messages).content

    # ── JSON extraction helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Robustly extract the first valid JSON object/array from model output.

        TinyLlama and other small models often wrap valid JSON in:
          - Prose:          "Sure! Here is the JSON: {...}"
          - Markdown fence: ```json\n{...}\n```
          - Mixed text:     "The answer is {...} Hope that helps!"

        Strategy (in order):
          1. Already valid JSON → return as-is
          2. Markdown code fence (```json...``` or ```...```)
          3. First {...} or [...] block in the text
          4. Return original so DeepEval can surface a readable error
        """
        stripped = text.strip()

        # 1. Already valid JSON
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        # 2. Markdown fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", stripped
        )
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 3. Raw JSON object or array anywhere in the text
        brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", stripped)
        if brace_match:
            candidate = brace_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 4. Fallback — return original; DeepEval error will show actual content
        return text


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def local_model() -> OllamaDeepEvalModel:
    """
    Single OllamaDeepEvalModel instance shared across the entire test session.
    Avoids re-creating ChatOllama connections for every test.
    """
    return OllamaDeepEvalModel(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
