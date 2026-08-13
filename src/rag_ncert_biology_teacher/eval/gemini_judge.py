"""The LLM DeepEval uses to SCORE answers (a "judge"), separate from the
Gemini model that actually generates them in rag/chain.py.

WHY Gemini as judge, not OpenAI (DeepEval's default): this project's chat/
captioning/embedding models are all Gemini, via the free Developer API
(config.py) - DeepEval ships a native GeminiModel that accepts a plain
api_key, the exact same auth already used everywhere else in this project,
so evaluation doesn't need a second provider or a separate credential.
"""

from deepeval.models import GeminiModel

from rag_ncert_biology_teacher.config import GEMINI_API_KEY, LLM_MODEL


def get_judge_model() -> GeminiModel:
    """A DeepEval-compatible Gemini model, authenticated the same way as
    every other Google API call in this project.
    """
    return GeminiModel(model=LLM_MODEL, api_key=GEMINI_API_KEY)
