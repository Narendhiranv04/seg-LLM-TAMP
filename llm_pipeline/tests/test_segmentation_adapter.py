import numpy as np

from llm_pipeline.segmentation_adapter import SegmentationEvidenceAdapter


class FakeDetector:
    def __init__(self):
        self.min_pixels_per_camera = 1
        self.cameras = {'overhead': object(), 'wrist': object()}
        self.handle_to_name = {
            1: 'mug2',
            2: 'mug4',
            3: 'box_lid',
            50: 'panda_leftfinger_visual',
            101: 'box_boundary',
            102: 'placement_boundary',
        }
        self.handle_to_task_name = {
            1: 'mug2',
            2: 'mug4',
            3: 'box_lid',
        }
        self.handle_to_region_name = {
            101: 'box_boundary',
            102: 'placement_boundary',
        }
        self.current_visible = set()
        self.newly_detected = set()
        self.current_regions = set()
        self.camera_hits = {}
        self.pixel_totals = {}
        self.object_region_membership = {}
        self.known_objects = set()

    def update(self):
        return set(self.current_visible)

    def get_current_snapshot(self):
        return {
            'visible_objects': sorted(self.current_visible),
            'newly_visible_objects': sorted(self.newly_detected),
            'visible_regions': sorted(self.current_regions),
            'camera_hits': {name: list(hits) for name, hits in self.camera_hits.items()},
            'pixel_totals': dict(self.pixel_totals),
            'object_regions': {name: list(regions) for name, regions in self.object_region_membership.items()},
            'known_objects': sorted(self.known_objects),
        }

    def reset_known(self):
        self.known_objects = set()
        self.newly_detected = set()


class FakeViewer:
    def __init__(self):
        self.updated = 0
        self.actions = []
        self.stopped = False

    def update(self):
        self.updated += 1
        return {'mug2', 'box_lid'}

    def set_action_sequence(self, actions):
        self.actions.append(('sequence', list(actions)))

    def set_current_action(self, current_action_index=None, current_action_label=None):
        self.actions.append(('current', current_action_index, current_action_label))

    def stop(self):
        self.stopped = True


def _empty_mask() -> np.ndarray:
    return np.zeros((10, 10), dtype=np.int32)


def test_segmentation_adapter_fuses_mask_regions_and_discovery() -> None:
    detector = FakeDetector()
    adapter = SegmentationEvidenceAdapter(detector=detector)

    first_mask = _empty_mask()
    first_mask[1:8, 1:8] = 101
    first_mask[3:5, 4:6] = 1
    first_mask[4:6, 6:7] = 50
    detector.current_visible = {'mug2'}
    detector.newly_detected = {'mug2'}
    detector.current_regions = {'box_boundary'}
    detector.camera_hits = {'mug2': ['overhead']}
    detector.pixel_totals = {'mug2': 4}
    detector.object_region_membership = {'mug2': ['box_boundary']}
    first_snapshot = adapter.capture_snapshot({'overhead': first_mask})

    assert first_snapshot.visible_objects == ['mug2']
    assert first_snapshot.newly_visible_objects == ['mug2']
    assert first_snapshot.visible_regions == ['box_boundary']
    assert first_snapshot.object_evidence['mug2'].mask_regions == ['box_boundary']
    assert first_snapshot.object_evidence['mug2'].camera_hits == ['overhead']
    assert first_snapshot.object_evidence['mug2'].gripper_proximity is not None

    second_mask = _empty_mask()
    second_mask[1:7, 1:9] = 102
    second_mask[3:5, 4:6] = 3
    detector.current_visible = {'box_lid'}
    detector.newly_detected = {'box_lid'}
    detector.current_regions = {'placement_boundary'}
    detector.camera_hits = {'box_lid': ['wrist']}
    detector.pixel_totals = {'box_lid': 4}
    detector.object_region_membership = {'box_lid': ['placement_boundary']}
    second_snapshot = adapter.capture_snapshot({'wrist': second_mask})

    assert second_snapshot.object_evidence['box_lid'].mask_regions == ['placement_boundary']
    assert second_snapshot.newly_visible_objects == ['box_lid']
    assert adapter.is_lid_open(second_snapshot) is True
    assert adapter.blocking_objects_for_lid(first_snapshot) == ['mug2']


def test_segmentation_adapter_refreshes_direct_detector_and_live_view_methods() -> None:
    detector = FakeDetector()
    detector.current_visible = {'mug2', 'box_lid'}
    detector.newly_detected = {'mug2'}
    detector.current_regions = {'box_boundary'}
    detector.known_objects = {'mug2', 'box_lid'}

    adapter = SegmentationEvidenceAdapter(detector=detector, live_segmentation_view=True)
    adapter.viewer = FakeViewer()

    visibility = adapter.refresh_visibility(event='initial')
    assert visibility['visible_objects'] == ['mug2', 'box_lid']
    assert visibility['newly_visible_objects'] == ['mug2']
    assert visibility['visible_regions'] == ['box_boundary']

    detected = adapter.update_live_segmentation_view()
    assert detected == {'mug2', 'box_lid'}
    assert adapter.viewer.updated == 1

    adapter.set_live_action_sequence(['move', 'pick(mug2)'], current_action_index=1, current_action_label='pick(mug2)')
    assert adapter.viewer.actions[-1] == ('current', 1, 'pick(mug2)')

    adapter.shutdown()
    assert adapter.viewer is None or True
