#!/usr/bin/env python3
"""Unit tests for cantonese_viseme.py (text-driven Cantonese lip shape modulation)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cantonese_viseme import (
    apply_viseme_mouth_warp,
    build_viseme_trajectory,
    classify_char_viseme,
)


class TestCantoneseViseme(unittest.TestCase):
    def test_char_classification(self):
        # Rounded vowels narrow horizontal width
        _, w_round, _ = classify_char_viseme("澳")
        self.assertLess(w_round, 0.96)
        # Spread vowels widen horizontal width
        _, w_spread, _ = classify_char_viseme("議")
        self.assertGreater(w_spread, 1.03)
        # Open vowels increase vertical opening
        o_open, _, _ = classify_char_viseme("大")
        self.assertGreater(o_open, 1.06)
        # Bilabial onset triggers closure flag
        _, _, is_bilab = classify_char_viseme("萬")
        self.assertTrue(is_bilab)

    def test_trajectory_gates_silence(self):
        sw = np.zeros(100, dtype=np.float32)
        sw[20:70] = 1.0
        o, w = build_viseme_trajectory(
            100, 25, sw, subtitle_text="訪澳國際旅客增長3.4%"
        )
        # Pure silence frames at start/end remain strictly neutral 1.0
        self.assertAlmostEqual(float(o[0]), 1.0, places=3)
        self.assertAlmostEqual(float(w[0]), 1.0, places=3)
        self.assertAlmostEqual(float(o[-1]), 1.0, places=3)
        # Active speech modulates smoothly
        self.assertGreater(float(o[20:70].max() - o[20:70].min()), 0.03)
        self.assertGreater(float(w[20:70].max() - w[20:70].min()), 0.03)

    def test_timeline_json_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl = Path(tmp) / "sub.json"
            tl.write_text(
                json.dumps([
                    {"start": 0.0, "end": 1.2, "text": "澳門旅遊"},
                    {"start": 1.4, "end": 2.4, "text": "議員表示"},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            sw = np.ones(65, dtype=np.float32)
            o, w = build_viseme_trajectory(65, 25, sw, subtitle_timeline=str(tl))
            # First segment (rounded vowels) should be narrower on average than second (spread vowels)
            self.assertLess(float(w[5:25].mean()), float(w[38:55].mean()))

    def test_mouth_warp_preserves_border(self):
        patch = np.zeros((120, 120, 3), dtype=np.uint8)
        patch[:, :, 1] = 200
        patch[0, :] = 255
        patch[-1, :] = 255
        warped = apply_viseme_mouth_warp(patch, (60.0, 72.0), 36.0, 18.0, 1.08, 0.94)
        self.assertEqual(warped.shape, patch.shape)
        self.assertTrue(np.array_equal(warped[0, :], patch[0, :]))
        self.assertTrue(np.array_equal(warped[-1, :], patch[-1, :]))


if __name__ == "__main__":
    unittest.main()
