from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from app import detect_smoking, parse_args, detect_video
from detector import OnnxDetector


class DetectionTests(unittest.TestCase):
    def test_negative_class_filtered_and_letterbox_coordinates_restored(self):
        class FakeSession:
            def run(self, _, inputs):
                self.input_shape = inputs['images'].shape
                return [np.array([
                    [320, 320, 64, 64, .9, .1],
                    [321, 321, 64, 64, .8, .1],
                    [100, 100, 50, 50, .01, .99],
                ], dtype=np.float32).T[None]]
        model = object.__new__(OnnxDetector)
        model.session = FakeSession()
        model.names = {0: 'Smooking', 1: 'not_smoking'}
        model.input_name = 'images'
        model.height = model.width = 640
        model.positive_ids = {0}
        model.confidence, model.iou = .5, .45
        result = model.predict(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(model.session.input_shape, (1, 3, 640, 640))
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]['bbox_xyxy'], [576, 296, 704, 424])

    def test_tiled_boxes_are_offset_to_original_image_and_deduplicated(self):
        class FakeDetector:
            confidence, iou = .5, .45
            def predict(self, frame):
                if frame.shape[:2] == (100, 100):
                    return [{'confidence': .9, 'bbox_xyxy': [45, 45, 55, 55]}]
                x, y = int(frame[0, 0, 0]), int(frame[0, 0, 1])
                return [{'confidence': .8, 'bbox_xyxy': [45-x, 45-y, 55-x, 55-y]}]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(100)[None, :]
        frame[:, :, 1] = np.arange(100)[:, None]
        result = detect_smoking(FakeDetector(), frame, enhance=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_xyxy'], [45, 45, 55, 55])

    def test_disabling_tiles_calls_detector_once(self):
        class FakeDetector:
            calls = 0
            def predict(self, frame):
                self.calls += 1
                return []
        model = FakeDetector()
        self.assertEqual(detect_smoking(model, np.zeros((20, 20, 3)), False), [])
        self.assertEqual(model.calls, 1)

    def test_cli_controls(self):
        args = parse_args(['sample.mp4', '--headless', '--max-frames', '3', '--no-tiles'])
        self.assertEqual(args.source, 'sample.mp4')
        self.assertEqual(args.max_frames, 3)
        self.assertTrue(args.headless)
        self.assertTrue(args.no_tiles)

    def test_invalid_inputs_fail_before_opening_camera(self):
        with self.assertRaises(ValueError):
            detect_video('')
        with self.assertRaises(ValueError):
            detect_video('sample.mp4', max_frames=-1)
        with self.assertRaises(ValueError):
            OnnxDetector(Path('missing.onnx'), ['Smooking'], 2, .45)

    def test_missing_weights_gives_download_instruction(self):
        with patch('detector.Path.is_file', return_value=False):
            with self.assertRaisesRegex(ValueError, 'download_model.py'):
                OnnxDetector(Path('missing.onnx'), ['Smooking'], .5, .45)


if __name__ == '__main__':
    unittest.main()
