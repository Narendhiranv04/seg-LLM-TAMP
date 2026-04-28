"""Canonical evaluation variants and benchmark defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    task_family: str
    scene_path: str
    action_sequence_length: Optional[int]
    gt_total_subtasks: Optional[int]
    gt_runner_path: Optional[str]
    goal_text: str
    expected_subtask_buckets: Dict[str, int] = field(default_factory=dict)
    model_eval_supported: bool = False
    model_eval_reason: str = ""
    pending: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            'variant_id': self.variant_id,
            'task_family': self.task_family,
            'scene_path': self.scene_path,
            'action_sequence_length': self.action_sequence_length,
            'gt_total_subtasks': self.gt_total_subtasks,
            'gt_runner_path': self.gt_runner_path,
            'goal_text': self.goal_text,
            'expected_subtask_buckets': dict(self.expected_subtask_buckets),
            'model_eval_supported': self.model_eval_supported,
            'model_eval_reason': self.model_eval_reason,
            'pending': self.pending,
        }


KITCHEN_GOAL_K1 = 'move all the groceries inside the cupboard and all mugs inside the box'
KITCHEN_GOAL_K2 = 'move all the groceries inside the cupboard and all mugs inside the box'
KITCHEN_GOAL_K3 = 'move all the groceries inside the cupboard and all mugs inside the box'
GRILL_GOAL_G1 = (
    'Open the grill, move the spam from inside the grill onto the table, move the outside meat onto the grill, '
    'close the grill, place the plate on the plate boundary, reopen the grill, and move the grilled meat onto the plate.'
)
GRILL_GOAL_G2 = (
    'Open the grill, place the plate on the plate boundary, move the inside meat onto the plate, '
    'move both outside meats onto the grill one by one, close the grill, reopen it, and move both grilled meats onto the plate.'
)
GRILL_GOAL_G3 = (
    'Open the grill, move the spam from inside the grill onto the table, place the plate on the plate boundary, '
    'move the remaining inside meat onto the plate, move both outside meats onto the grill one by one, '
    'close the grill, reopen it, and move both grilled meats onto the plate.'
)


VARIANTS: Dict[str, VariantSpec] = {
    'K1': VariantSpec(
        variant_id='K1',
        task_family='kitchen',
        scene_path=str(ROOT_DIR / 'task1_variation1.ttt'),
        action_sequence_length=35,
        gt_total_subtasks=7,
        gt_runner_path=str(ROOT_DIR / 'variation_1_easy' / 'ground_truth_orchestrator_variation1_easy.py'),
        goal_text=KITCHEN_GOAL_K1,
        expected_subtask_buckets={
            'mug_to_placement': 2,
            'open_lid': 1,
            'grocery_to_cupboard': 2,
            'mug_to_box': 2,
        },
        model_eval_supported=True,
    ),
    'K2': VariantSpec(
        variant_id='K2',
        task_family='kitchen',
        scene_path=str(ROOT_DIR / 'task1_variation2.ttt'),
        action_sequence_length=35,
        gt_total_subtasks=7,
        gt_runner_path=str(ROOT_DIR / 'variation_2' / 'ground_truth_orchestrator_variation2.py'),
        goal_text=KITCHEN_GOAL_K2,
        expected_subtask_buckets={
            'mug_to_placement': 2,
            'open_lid': 1,
            'grocery_to_cupboard': 2,
            'mug_to_box': 2,
        },
        model_eval_supported=True,
    ),
    'K3': VariantSpec(
        variant_id='K3',
        task_family='kitchen',
        scene_path=str(ROOT_DIR / 'task1_variation3.ttt'),
        action_sequence_length=40,
        gt_total_subtasks=8,
        gt_runner_path=str(ROOT_DIR / 'variation_3_hard' / 'ground_truth_orchestrator_variation3_hard.py'),
        goal_text=KITCHEN_GOAL_K3,
        expected_subtask_buckets={
            'mug_to_placement': 2,
            'open_lid': 1,
            'grocery_to_cupboard': 2,
            'mug_to_box': 3,
        },
        model_eval_supported=True,
    ),
    'G1': VariantSpec(
        variant_id='G1',
        task_family='grill',
        scene_path=str(ROOT_DIR / 'grill_task2' / 'grill.variation1.ttt'),
        action_sequence_length=35,
        gt_total_subtasks=7,
        gt_runner_path=str(ROOT_DIR / 'grill_task2' / 'ground_truth_orchestrator_variation1 copy.py'),
        goal_text=GRILL_GOAL_G1,
        expected_subtask_buckets={
            'open_grill': 2,
            'close_grill': 1,
            'plate_to_boundary': 1,
            'meat_to_plate': 1,
            'meat_to_grill': 1,
            'meat_to_table': 1,
        },
        model_eval_supported=False,
        model_eval_reason='No grill VLM/LLM replanning runner exists in the current repo.',
    ),
    'G2': VariantSpec(
        variant_id='G2',
        task_family='grill',
        scene_path=str(ROOT_DIR / 'grill_task2' / 'grill.variation2.ttt'),
        action_sequence_length=45,
        gt_total_subtasks=9,
        gt_runner_path=str(ROOT_DIR / 'grill_task2' / 'ground_truth_orchestrator_variation1 copy.py'),
        goal_text=GRILL_GOAL_G2,
        expected_subtask_buckets={
            'open_grill': 2,
            'close_grill': 1,
            'plate_to_boundary': 1,
            'meat_to_plate': 3,
            'meat_to_grill': 2,
        },
        model_eval_supported=False,
        model_eval_reason='No grill VLM/LLM replanning runner exists in the current repo.',
    ),
    'G3': VariantSpec(
        variant_id='G3',
        task_family='grill',
        scene_path=str(ROOT_DIR / 'grill_task2' / 'grill.variation3.ttt'),
        action_sequence_length=50,
        gt_total_subtasks=10,
        gt_runner_path=str(ROOT_DIR / 'grill_task2' / 'ground_truth_orchestrator_variation1 copy.py'),
        goal_text=GRILL_GOAL_G3,
        expected_subtask_buckets={
            'open_grill': 2,
            'close_grill': 1,
            'plate_to_boundary': 1,
            'meat_to_plate': 3,
            'meat_to_grill': 2,
            'meat_to_table': 1,
        },
        model_eval_supported=False,
        model_eval_reason='No grill VLM/LLM replanning runner exists in the current repo.',
    ),
}


DEFAULT_GT_VARIANTS = ['K1', 'K2', 'K3', 'G1', 'G2', 'G3']
DEFAULT_MODEL_VARIANTS = ['K1', 'K2', 'K3']
DEFAULT_VLM_MODEL_ALIASES = ['qwen-vl', 'gsarch', 'ms-phi4', 'internvl-3.5']
DEFAULT_LLM_MODEL_ALIASES = ['qwen', 'selene', 'deepseek-r1', 'mistral-nemo']


def get_variant_spec(variant_id: str) -> VariantSpec:
    key = (variant_id or '').strip().upper()
    if key not in VARIANTS:
        raise KeyError(f'Unknown variant: {variant_id}')
    return VARIANTS[key]
