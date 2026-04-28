"""Evaluation helpers for multi-model benchmarking."""

from evaluation.canonical_variants import (
    DEFAULT_GT_VARIANTS,
    DEFAULT_LLM_MODEL_ALIASES,
    DEFAULT_MODEL_VARIANTS,
    DEFAULT_VLM_MODEL_ALIASES,
    VARIANTS,
    VariantSpec,
    get_variant_spec,
)
from evaluation.metrics import (
    aggregate_gt_records,
    aggregate_model_records,
    classify_failure_message,
    collect_failure_occurrences,
    score_variant_completion,
)

__all__ = [
    'DEFAULT_GT_VARIANTS',
    'DEFAULT_LLM_MODEL_ALIASES',
    'DEFAULT_MODEL_VARIANTS',
    'DEFAULT_VLM_MODEL_ALIASES',
    'VARIANTS',
    'VariantSpec',
    'aggregate_gt_records',
    'aggregate_model_records',
    'classify_failure_message',
    'collect_failure_occurrences',
    'get_variant_spec',
    'score_variant_completion',
]
