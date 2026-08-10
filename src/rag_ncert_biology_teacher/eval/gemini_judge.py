"""The LLM DeepEval uses to SCORE answers (a "judge"), separate from the
Gemini model that actually generates them in rag/chain.py.

WHY Gemini as judge, not OpenAI (DeepEval's default): the whole project runs
on GCP/Vertex AI (Step 1) - DeepEval ships a native GeminiModel that works
with Vertex AI + Application Default Credentials, the exact same auth
already used everywhere else in this project, so evaluation doesn't need a
second provider or API key just to judge answers.
"""

from deepeval.models import GeminiModel

from rag_ncert_biology_teacher.config import GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_PROJECT, LLM_MODEL


def get_judge_model() -> GeminiModel:
    """A DeepEval-compatible Gemini model, authenticated the same way as
    every other Google API call in this project.
    """
    return GeminiModel(
        model=LLM_MODEL,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        use_vertexai=True,
    )
