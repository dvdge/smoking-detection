import unittest
from app import DetectionMemory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.item = [{"confidence": 0.8, "bbox_xyxy": [10, 20, 30, 40]}]

    def test_unseen_target_cannot_be_invented(self):
        self.assertEqual(DetectionMemory(.6).update([], 0), ([], False, 0.0))

    def test_short_gap_is_marked_as_history_and_expires(self):
        memory = DetectionMemory(.6)
        self.assertFalse(memory.update(self.item, 0)[1])
        displayed, remembered, age = memory.update([], .2)
        self.assertEqual(displayed, self.item)
        self.assertTrue(remembered)
        self.assertAlmostEqual(age, .2)
        self.assertTrue(memory.update([], .5)[1])
        self.assertEqual(memory.update([], .7), ([], False, 0.0))
        self.assertEqual(memory.update([], 1.0), ([], False, 0.0))

    def test_reappearing_target_is_a_fresh_detection(self):
        memory = DetectionMemory(.6)
        memory.update(self.item, 0)
        memory.update([], .2)
        self.assertEqual(memory.update(self.item, .3), (self.item, False, 0.0))
        self.assertTrue(memory.update([], .8)[1])

    def test_clock_reversal_does_not_keep_future_evidence(self):
        memory = DetectionMemory(.6)
        memory.update(self.item, 5)
        self.assertEqual(memory.update([], 1), ([], False, 0.0))


if __name__ == '__main__':
    unittest.main()
