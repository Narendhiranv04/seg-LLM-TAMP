import sys
import types


pyrep_module = types.ModuleType('pyrep')
backend_module = types.ModuleType('pyrep.backend')
backend_module.sim = object()
objects_module = types.ModuleType('pyrep.objects')
vision_sensor_module = types.ModuleType('pyrep.objects.vision_sensor')
class MissingVisionSensor:
    def __init__(self, name):
        raise RuntimeError(name)


vision_sensor_module.VisionSensor = MissingVisionSensor
sys.modules.setdefault('pyrep', pyrep_module)
sys.modules.setdefault('pyrep.backend', backend_module)
sys.modules.setdefault('pyrep.objects', objects_module)
sys.modules.setdefault('pyrep.objects.vision_sensor', vision_sensor_module)

from segmentation_object_detector import SegmentationObjectDetector


def test_segmentation_detector_canonicalizes_grill_scene_names() -> None:
    detector = SegmentationObjectDetector.__new__(SegmentationObjectDetector)

    assert detector._canonical_task_name('steak_visual') == 'steak'
    assert detector._canonical_task_name('chicken_visual') == 'chicken'
    assert detector._canonical_task_name('plate_visual') == 'plate'
    assert detector._canonical_task_name('lid_visual') == 'grill_lid'
    assert detector._canonical_task_name('plate_boundary') is None


def test_segmentation_detector_task_objects_include_env_names() -> None:
    detector = SegmentationObjectDetector.__new__(SegmentationObjectDetector)
    detector.env = type(
        'FakeEnv',
        (),
        {
            'name_to_obj': {
                'steak': object(),
                'chicken': object(),
                'plate': object(),
                'grill_lid': object(),
            }
        },
    )()

    default_task_objects = {
        'mug1', 'mug2', 'mug3', 'mug4',
        'soup', 'mustard', 'spam', 'sugar', 'crackers',
        'box_lid', 'cupboard',
    }
    env_task_objects = {
        name
        for name, obj in detector.env.name_to_obj.items()
        if name and obj is not None
    }
    detector.task_objects = default_task_objects | env_task_objects

    assert {'steak', 'chicken', 'plate', 'grill_lid'} <= detector.task_objects


def test_segmentation_detector_falls_back_to_full_env_camera_names() -> None:
    detector = SegmentationObjectDetector.__new__(SegmentationObjectDetector)
    detector.cameras = {}
    detector.env = type(
        'FakeEnv',
        (),
        {
            'cams': {
                'cam_over_shoulder_left': 'left-cam',
                'cam_over_shoulder_right': 'right-cam',
                'cam_overhead': 'overhead-cam',
                'cam_wrist': 'wrist-cam',
                'cam_front': 'front-cam',
            }
        },
    )()

    detector._setup_cameras()

    assert detector.cameras == {
        'left': ('render_mode', 'left-cam'),
        'right': ('render_mode', 'right-cam'),
        'overhead': ('render_mode', 'overhead-cam'),
        'wrist': ('render_mode', 'wrist-cam'),
        'front': ('render_mode', 'front-cam'),
    }
