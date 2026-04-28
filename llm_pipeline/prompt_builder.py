"""Text-only prompt bundle and prompt rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from llm_pipeline.executable_symbols import RuntimeSymbolRegistry, build_runtime_symbol_registry
from llm_pipeline.pipeline_types import (
    BaseContextBuilder,
    ICLMode,
    PromptBundle,
    SceneState,
    SegmentationSnapshot,
    TextPromptBundle,
)


PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'


class TextOnlyContextBuilder(BaseContextBuilder):
    """Builds text-only prompts from segmentation-derived observation state."""

    def __init__(self, env=None, symbol_registry: Optional[RuntimeSymbolRegistry] = None):
        self.env = env
        self.symbol_registry = symbol_registry or build_runtime_symbol_registry(env=env)

    def set_env(self, env) -> None:
        self.env = env

    def set_symbol_registry(self, symbol_registry: RuntimeSymbolRegistry) -> None:
        self.symbol_registry = symbol_registry

    def build_bundle(
        self,
        state: SceneState,
        goal_text: str,
        failure_event: Optional[Any] = None,
        previous_actions: Optional[Iterable[str]] = None,
        icl_mode: str = ICLMode.ZERO_SHOT.value,
    ) -> PromptBundle:
        # 1. Reconstruct snapshot-like data from SceneState for the text builder logic
        # In a more refined version, we would refactor _build_observation_text to take SceneState.
        # For now, we use the fact that SceneState was built from a snapshot in pipeline.py.
        snapshot = getattr(state, '_original_snapshot', None)
        held_object = state.gripper_state.get('holding')

        observation_text = self._build_observation_text(snapshot=snapshot, held_object=held_object)
        visible_text = self._build_visible_text(snapshot=snapshot)

        # 2. Create the intermediate text bundle
        text_bundle = TextPromptBundle(
            goal_text=goal_text,
            observation_text=observation_text,
            visible_objects_text=visible_text,
            failure_context=failure_event.message if failure_event else None,
            icl_mode=icl_mode,
            previous_actions=tuple(previous_actions or ()),
        )

        # 3. Render final prompts
        system_prompt = self.build_system_prompt(text_bundle)
        user_prompt = self.build_user_prompt(text_bundle)

        return PromptBundle(
            goal_text=goal_text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            visible_objects=state.visible_objects,
            valid_regions=state.valid_regions,
            icl_mode=icl_mode,
            previous_actions=text_bundle.previous_actions,
            failure_context=text_bundle.failure_context,
        )

    def build_system_prompt(self, bundle: TextPromptBundle) -> str:
        base = (PROMPTS_DIR / 'system_prompt.txt').read_text(encoding='utf-8').strip()
        if bundle.icl_mode == ICLMode.FEW_SHOT_SHARED_1.value:
            example = (PROMPTS_DIR / 'shared_exemplar.txt').read_text(encoding='utf-8').strip()
            return f"{base}\n\nSHARED FEW-SHOT EXEMPLAR:\n{example}\n"
        return f"{base}\n"

    def build_user_prompt(self, bundle: TextPromptBundle) -> str:
        parts: List[str] = [
            bundle.observation_text,
            '',
            bundle.visible_objects_text,
            '',
            "GOAL:",
            bundle.goal_text,
        ]
        if bundle.previous_actions:
            # Strip move annotations (move(→pick) → move) so the LLM sees bare actions
            cleaned = [
                'move' if str(a).startswith('move(') and '→' in str(a) else str(a)
                for a in bundle.previous_actions
            ]
            parts.extend(['', 'PREVIOUS ACTIONS (already executed, do not repeat):', *cleaned])
        if bundle.failure_context:
            parts.extend(['', 'FAILURE CONTEXT:', bundle.failure_context])
        parts.extend(['', *self._build_action_contract_lines()])
        return '\n'.join(parts).strip()

    def _build_action_contract_lines(self) -> List[str]:
        lines = ['OUTPUT CONTRACT:']
        if self.symbol_registry.actions:
            lines.append('available_actions=' + ', '.join(self.symbol_registry.actions))
        lines.append('Use only object and region names that appear in the observation above.')
        lines.append('Executable action formats for this run:')
        for action_name in self.symbol_registry.actions:
            lines.append(self._action_format_line(action_name))
        lines.append('Return executable action lines only.')
        return lines

    def _action_format_line(self, action_name: str) -> str:
        if action_name == 'move':
            return 'move'
        if action_name == 'pick':
            return 'pick(object_name)'
        if action_name == 'place':
            return 'place(object_name, region_name)'
        if action_name == 'open':
            if 'box_lid' in self.symbol_registry.objects:
                return 'open(box_lid)'
            return 'open(object_name)'
        return f'{action_name}(...)'

    def _build_observation_text(
        self,
        snapshot: Optional[SegmentationSnapshot],
        held_object: Optional[str],
    ) -> str:
        visible_objects = list(snapshot.visible_objects) if snapshot is not None else []
        visible_regions = list(snapshot.visible_regions) if snapshot is not None else []
        supported_regions = list(snapshot.supported_regions) if snapshot is not None else list(self.symbol_registry.regions)

        lines = ['CURRENT SEGMENTATION SNAPSHOT:', '']
        lines.append(f'- frame_index: {snapshot.frame_index if snapshot is not None else 0}')
        lines.append(f'- held_object: {held_object or "none"}')
        lines.append(f'- visible_objects: {", ".join(visible_objects) if visible_objects else "(none)"}')
        lines.append('- newly_visible_objects: ' + (', '.join(snapshot.newly_visible_objects) if snapshot is not None and snapshot.newly_visible_objects else '(none)'))
        lines.append(f'- visible_regions: {", ".join(visible_regions) if visible_regions else "(none)"}')
        lines.append(f'- supported_regions: {", ".join(supported_regions) if supported_regions else "(none)"}')
        if snapshot is not None:
            gripper_visible = bool(snapshot.gripper_evidence.get('visible'))
            lines.append(f'- gripper_mask_visible: {str(gripper_visible).lower()}')

        lines.extend(['', 'VISIBLE OBJECT EVIDENCE:'])
        if not visible_objects:
            lines.append('- (none)')
            return '\n'.join(lines)

        for name in visible_objects:
            evidence = snapshot.object_evidence.get(name) if snapshot is not None else None
            if evidence is None:
                lines.append(f'- {name}: visible=true')
                continue
            facts = []
            if evidence.camera_hits:
                facts.append(f'camera_hits={"|".join(evidence.camera_hits)}')
            if evidence.camera_pixels:
                facts.append(f'camera_pixels={self._format_camera_pixels(evidence.camera_pixels)}')
            if evidence.pixel_count:
                facts.append(f'pixel_count={evidence.pixel_count}')
            if evidence.mask_regions:
                facts.append(f'mask_regions={"|".join(evidence.mask_regions)}')
            if evidence.centroid:
                facts.append(f'centroids={self._format_point_map(evidence.centroid)}')
            if evidence.bbox:
                facts.append(f'bboxes={self._format_bbox_map(evidence.bbox)}')
            if evidence.newly_visible:
                facts.append('newly_visible=true')
            if evidence.gripper_proximity is not None:
                facts.append(f'gripper_proximity={evidence.gripper_proximity:.4f}')
            lines.append(f'- {name}: {", ".join(facts) if facts else "visible=true"}')

        return '\n'.join(lines)

    def _build_visible_text(self, snapshot: Optional[SegmentationSnapshot]) -> str:
        visible_objects = list(snapshot.visible_objects) if snapshot is not None else []
        newly_visible = list(snapshot.newly_visible_objects) if snapshot is not None else []
        visible_regions = list(snapshot.visible_regions) if snapshot is not None else []
        lines = ['COMPACT SEGMENTATION SUMMARY:', '']
        lines.append('visible_objects=' + (', '.join(visible_objects) if visible_objects else '(none)'))
        lines.append('newly_visible_objects=' + (', '.join(newly_visible) if newly_visible else '(none)'))
        lines.append('visible_regions=' + (', '.join(visible_regions) if visible_regions else '(none)'))
        if snapshot is None:
            return '\n'.join(lines)
        for name in visible_objects:
            evidence = snapshot.object_evidence.get(name)
            if evidence is None:
                continue
            facts = []
            if evidence.mask_regions:
                facts.append(f'mask_regions={"|".join(evidence.mask_regions)}')
            if evidence.camera_hits:
                facts.append(f'camera_hits={"|".join(evidence.camera_hits)}')
            if evidence.pixel_count:
                facts.append(f'pixel_count={evidence.pixel_count}')
            if evidence.gripper_proximity is not None:
                facts.append(f'gripper_proximity={evidence.gripper_proximity:.4f}')
            lines.append(f'- {name}: {", ".join(facts) if facts else "visible=true"}')
        return '\n'.join(lines)

    @staticmethod
    def _format_camera_pixels(camera_pixels) -> str:
        return '|'.join(f'{name}:{int(count)}' for name, count in sorted(camera_pixels.items()))

    @staticmethod
    def _format_point_map(points) -> str:
        return '|'.join(
            f'{name}:({coords[0]:.4f},{coords[1]:.4f})'
            for name, coords in sorted(points.items())
        )

    @staticmethod
    def _format_bbox_map(boxes) -> str:
        return '|'.join(
            f'{name}:({coords[0]:.4f},{coords[1]:.4f},{coords[2]:.4f},{coords[3]:.4f})'
            for name, coords in sorted(boxes.items())
        )
