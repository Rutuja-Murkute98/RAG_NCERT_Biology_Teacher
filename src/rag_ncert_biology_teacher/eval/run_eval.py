"""Objectively score the RAG chain's answer quality with DeepEval, instead
of relying on eyeballing a handful of test questions.

WHY this step exists: everything before this point was validated by us
reading the output and judging "does this look right." That doesn't scale
and is easy to fool yourself with (an answer can sound confident and
plausible while being subtly unfaithful to the source). DeepEval runs each
answer through a set of standardized RAG metrics, each with its own scoring
rubric and threshold, giving an objective pass/fail plus a written reason.

Metrics used:
  - Faithfulness: does the answer only state things actually supported by
    the retrieved context? (the direct hallucination check)
  - Answer Relevancy: does the answer actually address the question asked,
    rather than wandering off-topic?
  - Contextual Precision: of the chunks retrieved, how many were actually
    relevant and correctly ranked? (needs a known-correct expected_output
    to judge against)
  - Contextual Recall: did retrieval find everything needed to fully answer,
    or miss something? (also needs expected_output)

DeepEval's default AsyncConfig runs up to 20 metric evaluations concurrently
-- 5 test cases x 4 metrics here is exactly 20, all hitting the SAME Gemini
judge model at once, and each metric can itself fire several internal
sub-calls (generate statements, generate verdicts, generate a reason),
pushing the real number of simultaneous calls well past what max_concurrent
alone controls (that setting only limits concurrency ACROSS test cases, not
within one). Lowering max_concurrent alone still timed out for real running
this exact evaluation, so async is disabled entirely below -- one LLM call
at a time is slower but reliable, the same "stay gentle on quota" tradeoff
already made everywhere else in this project (see retry_utils.py).

TEST_CASES below pairs each question with a hand-written expected_output --
a concise, accurate answer based on the actual NCERT content (verified
against this project's own earlier manual testing, see chain.py's docstring
and this project's build history) -- Contextual Precision/Recall can't run
without one.
"""

import sys
import time

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from rag_ncert_biology_teacher.eval.gemini_judge import get_judge_model
from rag_ncert_biology_teacher.rag.chain import ask

COOLDOWN_SECONDS = 60  # a full per-minute quota window, not a guess

TEST_CASES = [
    {
        "question": "How does tubal ligation work?",
        "expected_output": (
            "Tubal ligation (tubectomy) is a surgical sterilisation method for females. "
            "A part of the fallopian tube is removed or tied up, blocking the path of the "
            "egg so it cannot meet sperm, preventing conception."
        ),
    },
    {
        "question": "What causes AIDS and how does it spread?",
        "expected_output": (
            "AIDS is caused by HIV (Human Immunodeficiency Virus), a retrovirus. It spreads "
            "through sexual contact with an infected person, transfusion of contaminated "
            "blood/blood products, sharing infected needles (e.g. among intravenous drug "
            "users), and from an infected mother to her child during pregnancy, birth, or "
            "breastfeeding. It does not spread through casual physical contact."
        ),
    },
    {
        "question": "What is DNA fingerprinting used for?",
        "expected_output": (
            "DNA fingerprinting is used in forensic science to identify individuals, such as "
            "matching DNA found at a crime scene to a suspect, and more broadly to determine "
            "population and genetic diversity."
        ),
    },
    {
        "question": "What is natural selection?",
        "expected_output": (
            "Natural selection is Darwin's concept describing how, within a population, "
            "individuals with heritable characteristics that make them better suited to "
            "their environment survive and reproduce more successfully than others. Over "
            "generations, this makes those advantageous traits more common in the "
            "population, driving evolutionary change."
        ),
    },
    {
        "question": "How does a biogas plant work?",
        "expected_output": (
            "A biogas plant uses anaerobic bacteria (methanogens) to digest organic waste "
            "such as cattle dung and plant material in an oxygen-free tank. This anaerobic "
            "digestion produces biogas (mainly methane, used as fuel) and leaves behind "
            "slurry that can be used as fertiliser."
        ),
    },
]


def build_test_cases() -> list[LLMTestCase]:
    """Run each real question through the actual RAG chain (rag/chain.py) --
    these are genuine end-to-end results, not hand-crafted examples, so the
    evaluation reflects what a student actually gets.
    """
    cases = []
    for i, spec in enumerate(TEST_CASES):
        if i > 0:
            time.sleep(3)  # stay gentle on the per-minute quota, same lesson as everywhere else
        answer, chunks, _show_image = ask(spec["question"])
        cases.append(
            LLMTestCase(
                input=spec["question"],
                actual_output=answer,
                expected_output=spec["expected_output"],
                retrieval_context=[chunk.page_content for chunk in chunks],
            )
        )
    return cases


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.eval.run_eval
    sys.stdout.reconfigure(encoding="utf-8")

    judge = get_judge_model()
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]

    print(f"Running {len(TEST_CASES)} questions through the real RAG chain...\n")
    test_cases = build_test_cases()

    # Even fully sequential (run_async=False), all 4 metrics together still
    # exhausted the judge model's per-minute quota partway through (found for
    # real running this exact evaluation, twice, with different mitigations
    # each time). Running ONE METRIC across all test cases per evaluate()
    # call, with a real cooldown between metrics, bounds the burst size far
    # more tightly than any concurrency/timeout setting could -- the quota
    # window gets real time to clear between metrics instead of accumulating
    # pressure across all 4 back-to-back.
    for i, metric in enumerate(metrics):
        if i > 0:
            print(f"\n(cooling down {COOLDOWN_SECONDS}s before the next metric, to respect the per-minute quota)\n")
            time.sleep(COOLDOWN_SECONDS)

        print(f"--- Metric {i + 1}/{len(metrics)}: {metric.__class__.__name__} ---")
        for attempt in range(2):
            try:
                evaluate(test_cases, [metric], async_config=AsyncConfig(run_async=False))
                break
            except Exception as exc:
                if attempt == 0:
                    print(f"\n({metric.__class__.__name__} failed with {type(exc).__name__}: {exc}; "
                          f"waiting {COOLDOWN_SECONDS}s and retrying once)\n")
                    time.sleep(COOLDOWN_SECONDS)
                else:
                    raise
