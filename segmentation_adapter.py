"""Direct segmentation bridge for the maintained LLM pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from typing import Dict, Optional, Tuple

import numpy as np

from llm_pipeline.executable_symbols import RuntimeSymbolRegistry, build_runtime_symbol_registry
from llm_pipeline.types import SegmentationObjectEvidence, SegmentationSnapshot

try:
    from segmentation_object_detector import SegmentationObjectDetector
except Exception:  # pragma: no cover - runtime dependency
    SegmentationObjectDetector = None

try:
    from live_segmentation_viewer import LiveSegmentationViewer
except Exception:  # pragma: no cover - runtime dependency
    LiveSegmentationViewer = None

try:
    from tkinter_segmentation_viewer import TkinterSegmentationViewer
except Exception:  # pragma: no cover - runtime dependency
    TkinterSegmentationViewer = None


class SegmentationEvidenceAdapter:
    """Uses direct segmentation masks as the LLM pipeline source of truth."""

    def __init__(
        self,
        env=None,
        detector=None,
        live_segmentation_view: bool = False,
        visible_objects_only: bool = True,
        symbol_registry: Optional[RuntimeSymbolRegistry] = None,
    ):
        self.env = env
        self.detector = detector
        self.live_segmentation_view = bool(live_segmentation_view)
        self.visible_objects_only = bool(visible_objects_only)
        self.frame_index = 0
        self.known_visible = set()
        self.last_visibility_snapshot = {
            'visible_objects': [],
            'newly_visible_objects': [],
            'visible_regions': [],
        }
        self.symbol_registry = symbol_registry or build_runtime_symbol_registry(env=env)
        self.viewer = None
        self._viewer_warning_printed = False

        self._sync_runtime_dependencies()
        # Rebuild registry with mask-discovered objects now that detector exists
        if symbol_registry is None and self.detector is not None and hasattr(self.detector, 'task_objects'):
            self.symbol_registry = build_runtime_symbol_registry(env=env, detected_objects=self.detector.task_objects)

    def set_env(self, env) -> None:
        if self.viewer is not None:
            try:
                self.viewer.stop()
            except Exception:
                pass
            self.viewer = None
        self.env = env
        self.detector = None
        self._sync_runtime_dependencies()
        # Rebuild registry with mask-discovered objects from the new detector
        detected_objects = None
        if self.detector is not None and hasattr(self.detector, 'task_objects'):
            detected_objects = self.detector.task_objects
        self.symbol_registry = build_runtime_symbol_registry(env=env, detected_objects=detected_objects)

    def set_symbol_registry(self, symbol_registry: RuntimeSymbolRegistry) -> None:
        self.symbol_registry = symbol_registry

    def reset_tracking(self) -> None:
        self.frame_index = 0
        self.known_visible = set()
        self.last_visibility_snapshot = {
            'visible_objects': [],
            'newly_visible_objects': [],
            'visible_regions': [],
        }
        if self.detector is not None and hasattr(self.detector, 'reset_known'):
            try:
                self.detector.reset_known()
            except Exception:
                pass

    def refresh_visibility(self, event: str = '') -> Dict[str, list[str]]:
        del event
        valid_objects = set(self.symbol_registry.objects)
        valid_regions = set(self.symbol_registry.regions)

        if self.detector is not None and hasattr(self.detector, 'update'):
            current_visible = set(self.detector.update())
            detector_snapshot = self._get_detector_snapshot()
            visible_now = self._ordered_tokens(
                [name for name in detector_snapshot.get('visible_objects', current_visible) if name in valid_objects],
                self.symbol_registry.objects,
            )
            newly_visible = self._ordered_tokens(
                [name for name in detector_snapshot.get('newly_visible_objects', []) if name in valid_objects],
                self.symbol_registry.objects,
            )
            visible_regions = self._ordered_tokens(
                [name for name in detector_snapshot.get('visible_regions', []) if name in valid_regions],
                self.symbol_registry.regions,
            )
            snapshot = {
                'visible_objects': visible_now,
                'newly_visible_objects': newly_visible,
                'visible_regions': visible_regions,
            }
            self.last_visibility_snapshot = snapshot
            return snapshot

        return dict(self.last_visibility_snapshot)

    def update_live_segmentation_view(self):
        viewer = self._ensure_viewer_started()
        if viewer is None:
            return set()
        try:
            return viewer.update()
        except Exception:
            return set()

    def set_live_action_sequence(self, actions, current_action_index=None, current_action_label=None) -> None:
        viewer = self._ensure_viewer_started()
        if viewer is None:
            return
        try:
            viewer.set_action_sequence(actions)
            viewer.set_current_action(current_action_index, current_action_label)
        except Exception:
            pass

    def shutdown(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.stop()
            except Exception:
                pass
            self.viewer = None

    def capture_snapshot(
        self,
        masks_by_camera: Optional[Dict[str, np.ndarray]] = None,
        event: str = '',
    ) -> SegmentationSnapshot:
        self.frame_index += 1
        valid_objects = set(self.symbol_registry.objects)
        valid_regions = set(self.symbol_registry.regions)

        detector_snapshot = {}
        if masks_by_camera is None:
            self.refresh_visibility(event=event)
            detector_snapshot = self._get_detector_snapshot()
            masks_by_camera = self._capture_masks()

        object_pixels = defaultdict(int)
        camera_pixels = defaultdict(dict)
        object_hits = defaultdict(list)
        object_bbox = defaultdict(dict)
        object_centroid = defaultdict(dict)
        region_votes = defaultdict(Counter)
        visible_regions = set(name for name in detector_snapshot.get('visible_regions', []) if name in valid_regions)
        gripper_centroids = {}

        handle_map = getattr(self.detector, 'handle_to_task_name', {}) if self.detector is not None else {}
        region_map = getattr(self.detector, 'handle_to_region_name', {}) if self.detector is not None else {}
        gripper_handles = self._build_gripper_handle_set()

        for camera_name, handle_mask in (masks_by_camera or {}).items():
            if handle_mask is None:
                continue

            if gripper_handles:
                gripper_mask = np.isin(handle_mask, list(gripper_handles))
                if np.any(gripper_mask):
                    gripper_centroids[camera_name] = self._normalize_centroid(
                        self._mask_centroid(gripper_mask),
                        handle_mask.shape,
                    )

            object_stats = self._extract_mask_stats(handle_mask, handle_map)
            region_stats = self._extract_mask_stats(handle_mask, region_map)
            inferred_regions = self._infer_regions_from_camera(object_stats, region_stats)

            for obj_name, stats in object_stats.items():
                if obj_name not in valid_objects:
                    continue
                object_pixels[obj_name] += int(stats['pixels'])
                camera_pixels[obj_name][camera_name] = int(stats['pixels'])
                object_hits[obj_name].append(camera_name)
                object_bbox[obj_name][camera_name] = self._normalize_bbox(stats['bbox'], handle_mask.shape)
                object_centroid[obj_name][camera_name] = self._normalize_centroid(stats['centroid'], handle_mask.shape)

            for region_name in region_stats:
                if region_name in valid_regions:
                    visible_regions.add(region_name)

            for obj_name, regions in inferred_regions.items():
                if obj_name not in valid_objects:
                    continue
                vote_map = region_votes[obj_name]
                for rank, region_name in enumerate(regions[:4]):
                    if region_name not in valid_regions:
                        continue
                    vote_map[region_name] += max(0.5, 3.0 - rank)

        if detector_snapshot:
            visible_now = self._ordered_tokens(
                [name for name in detector_snapshot.get('visible_objects', []) if name in valid_objects],
                self.symbol_registry.objects,
            )
            newly_visible = self._ordered_tokens(
                [name for name in detector_snapshot.get('newly_visible_objects', []) if name in valid_objects],
                self.symbol_registry.objects,
            )
        else:
            visible_now = self._ordered_tokens(object_pixels.keys(), self.symbol_registry.objects)
            newly_visible = self._ordered_tokens(set(visible_now) - self.known_visible, self.symbol_registry.objects)

        self.known_visible.update(visible_now)

        object_evidence = {}
        detector_object_regions = detector_snapshot.get('object_regions', {}) if detector_snapshot else {}
        detector_camera_hits = detector_snapshot.get('camera_hits', {}) if detector_snapshot else {}
        detector_pixel_totals = detector_snapshot.get('pixel_totals', {}) if detector_snapshot else {}

        for object_name in visible_now:
            vote_items = sorted(
                region_votes.get(object_name, {}).items(),
                key=lambda item: (-item[1], self._region_rank(item[0]), item[0]),
            )
            ordered_regions = []
            seen_regions = set()
            for region_name in detector_object_regions.get(object_name, []):
                if region_name in valid_regions and region_name not in seen_regions:
                    seen_regions.add(region_name)
                    ordered_regions.append(region_name)
            for region_name, _ in vote_items:
                if region_name in valid_regions and region_name not in seen_regions:
                    seen_regions.add(region_name)
                    ordered_regions.append(region_name)

            gripper_proximity = self._compute_gripper_proximity(
                object_centroid.get(object_name, {}),
                gripper_centroids,
            )
            object_evidence[object_name] = SegmentationObjectEvidence(
                name=object_name,
                visible=True,
                camera_hits=sorted(set(object_hits.get(object_name, []) or detector_camera_hits.get(object_name, []))),
                pixel_count=int(object_pixels.get(object_name, detector_pixel_totals.get(object_name, 0))),
                camera_pixels=dict(camera_pixels.get(object_name, {})),
                bbox=dict(object_bbox.get(object_name, {})),
                centroid=dict(object_centroid.get(object_name, {})),
                mask_regions=ordered_regions,
                region_votes={name: float(score) for name, score in vote_items},
                newly_visible=object_name in set(newly_visible),
                gripper_proximity=gripper_proximity,
            )

        return SegmentationSnapshot(
            frame_index=self.frame_index,
            visible_objects=visible_now,
            newly_visible_objects=list(newly_visible),
            object_evidence=object_evidence,
            gripper_evidence={
                'camera_centroids': dict(gripper_centroids),
                'visible': bool(gripper_centroids),
            },
            supported_regions=list(self.symbol_registry.regions),
            visible_regions=self._ordered_tokens(visible_regions, self.symbol_registry.regions),
        )

    def is_lid_open(self, snapshot: SegmentationSnapshot) -> bool:
        lid_evidence = snapshot.object_evidence.get('box_lid')
        if lid_evidence is None or not lid_evidence.visible:
            return False
        mask_regions = set(lid_evidence.mask_regions)
        return bool(mask_regions) and 'box_boundary' not in mask_regions

    def blocking_objects_for_lid(self, snapshot: SegmentationSnapshot) -> list[str]:
        blockers = []
        for name, evidence in snapshot.object_evidence.items():
            if name == 'box_lid' or not evidence.visible:
                continue
            if 'box_boundary' in set(evidence.mask_regions):
                blockers.append(name)
        return sorted(blockers)

    def _sync_runtime_dependencies(self) -> None:
        if self.detector is None and self.env is not None and SegmentationObjectDetector is not None:
            self.detector = SegmentationObjectDetector(self.env)

    def _ensure_viewer_started(self):
        if self.viewer is not None:
            return self.viewer
        if not self.live_segmentation_view or self.env is None:
            return None

        backend_pref = os.environ.get('LIVE_SEG_VIEWER_BACKEND', 'auto').strip().lower()
        candidates = []
        if backend_pref == 'tkinter':
            candidates = [TkinterSegmentationViewer, LiveSegmentationViewer]
        elif backend_pref == 'opencv':
            candidates = [LiveSegmentationViewer, TkinterSegmentationViewer]
        else:
            candidates = [TkinterSegmentationViewer, LiveSegmentationViewer]

        for viewer_cls in candidates:
            if viewer_cls is None:
                continue
            try:
                viewer = viewer_cls(self.env)
                viewer.start()
                self.viewer = viewer
                return self.viewer
            except Exception as exc:
                if not self._viewer_warning_printed:
                    print(f'[SegmentationAdapter] Live viewer unavailable: {exc}')
                    self._viewer_warning_printed = True
        return None

    def _build_gripper_handle_set(self) -> set[int]:
        handles = set()
        if self.detector is None:
            return handles
        for handle, name in getattr(self.detector, 'handle_to_name', {}).items():
            lowered = (name or '').lower()
            if any(token in lowered for token in ('gripper', 'finger', 'panda_leftfinger', 'panda_rightfinger')):
                handles.add(int(handle))
        return handles

    def _capture_masks(self) -> Dict[str, np.ndarray]:
        masks = {}
        if self.detector is None:
            return masks
        capture = getattr(self.detector, '_capture_mask', None)
        if capture is None:
            return masks
        for camera_name in getattr(self.detector, 'cameras', {}):
            masks[camera_name] = capture(camera_name)
        return masks

    def _get_detector_snapshot(self) -> Dict[str, object]:
        if self.detector is None or not hasattr(self.detector, 'get_current_snapshot'):
            return {}
        try:
            return dict(self.detector.get_current_snapshot())
        except Exception:
            return {}

    def _extract_mask_stats(self, handle_mask, handle_to_label: Dict[int, str]):
        if self.detector is not None and hasattr(self.detector, '_extract_mask_stats'):
            try:
                return self.detector._extract_mask_stats(handle_mask, handle_to_label)
            except Exception:
                pass
        if handle_mask is None:
            return {}

        stats = {}
        unique_handles = np.unique(handle_mask)
        min_pixels = 10
        if self.detector is not None:
            min_pixels = int(getattr(self.detector, 'min_pixels_per_camera', 10))

        for handle in unique_handles:
            if int(handle) <= 0:
                continue
            label = handle_to_label.get(int(handle))
            if label is None:
                continue
            obj_mask = handle_mask == handle
            pixels = int(np.count_nonzero(obj_mask))
            if pixels < min_pixels:
                continue
            ys, xs = np.nonzero(obj_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            centroid = (float(xs.mean()), float(ys.mean()))
            entry = stats.get(label)
            if entry is None:
                stats[label] = {
                    'pixels': pixels,
                    'bbox': bbox,
                    'centroid_sum_x': centroid[0] * pixels,
                    'centroid_sum_y': centroid[1] * pixels,
                }
                continue
            entry['pixels'] += pixels
            bx0, by0, bx1, by1 = entry['bbox']
            entry['bbox'] = (
                min(bx0, bbox[0]),
                min(by0, bbox[1]),
                max(bx1, bbox[2]),
                max(by1, bbox[3]),
            )
            entry['centroid_sum_x'] += centroid[0] * pixels
            entry['centroid_sum_y'] += centroid[1] * pixels

        for label, entry in stats.items():
            pixels = max(1, int(entry['pixels']))
            entry['centroid'] = (
                float(entry['centroid_sum_x']) / pixels,
                float(entry['centroid_sum_y']) / pixels,
            )
            entry.pop('centroid_sum_x', None)
            entry.pop('centroid_sum_y', None)
        return stats

    def _infer_regions_from_camera(self, object_stats, region_stats):
        if self.detector is not None and hasattr(self.detector, '_infer_regions_from_camera'):
            try:
                return self.detector._infer_regions_from_camera(object_stats, region_stats)
            except Exception:
                pass
        inferred = {}
        for obj_name, obj_entry in object_stats.items():
            candidates = []
            for region_name, region_entry in region_stats.items():
                score = 0.0
                if self._bbox_contains_point(region_entry['bbox'], obj_entry['centroid'], pad=6):
                    score += 2.0
                score += self._bbox_overlap_ratio(obj_entry['bbox'], region_entry['bbox'])
                if score <= 0.0:
                    continue
                candidates.append((region_name, score))
            candidates.sort(key=lambda item: (-item[1], self._region_rank(item[0]), item[0]))
            inferred[obj_name] = [name for name, _ in candidates]
        return inferred

    def _mask_centroid(self, obj_mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(obj_mask)
        return (float(xs.mean()), float(ys.mean()))

    def _normalize_bbox(self, bbox, shape) -> Tuple[float, float, float, float]:
        h, w = shape[:2]
        x0, y0, x1, y1 = bbox
        return (
            float(x0 / max(1, w - 1)),
            float(y0 / max(1, h - 1)),
            float(x1 / max(1, w - 1)),
            float(y1 / max(1, h - 1)),
        )

    def _normalize_centroid(self, centroid, shape) -> Tuple[float, float]:
        h, w = shape[:2]
        x, y = centroid
        return (
            float(x / max(1, w - 1)),
            float(y / max(1, h - 1)),
        )

    @staticmethod
    def _bbox_contains_point(bbox, point, pad: int = 4) -> bool:
        x0, y0, x1, y1 = bbox
        px, py = point
        return (x0 - pad) <= px <= (x1 + pad) and (y0 - pad) <= py <= (y1 + pad)

    @staticmethod
    def _bbox_overlap_ratio(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0, min(ay1, by1) - max(ay0, by0))
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter = float(inter_w * inter_h)
        area = max(1.0, float((ax1 - ax0) * (ay1 - ay0)))
        return inter / area

    def _region_rank(self, region_name: str) -> int:
        try:
            return list(self.symbol_registry.regions).index(region_name)
        except ValueError:
            return len(self.symbol_registry.regions)

    @staticmethod
    def _ordered_tokens(tokens, reference_order) -> list[str]:
        reference = list(reference_order)
        seen = set()
        ordered = []
        for token in reference:
            if token in tokens and token not in seen:
                seen.add(token)
                ordered.append(token)
        for token in sorted(str(item) for item in tokens):
            if token not in seen:
                seen.add(token)
                ordered.append(token)
        return ordered

    @staticmethod
    def _compute_gripper_proximity(
        object_centroids: Dict[str, Tuple[float, float]],
        gripper_centroids: Dict[str, Tuple[float, float]],
    ) -> Optional[float]:
        best = None
        for camera_name, centroid in object_centroids.items():
            gripper = gripper_centroids.get(camera_name)
            if gripper is None:
                continue
            distance = float(np.linalg.norm(np.array(centroid) - np.array(gripper)))
            if best is None or distance < best:
                best = distance
        return best
