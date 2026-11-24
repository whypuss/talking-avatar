#!/usr/bin/env python3
"""Unit tests for whypv2.py (dual-image B-roll auto-window & CLI defaults)."""

import json
import tempfile
import unittest
from pathlib import Path

from whypv2 import auto_detect_broll_window, build_parser


class TestWhypv2(unittest.TestCase):
    def test_auto_broll_window_aligns_to_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl = Path(tmp) / "sub.json"
            tl.write_text(
                json.dumps([
                    {"start": 0.0, "end": 2.1, "text": "男子青洲疑墮樓送院"},
                    {"start": 2.1, "end": 7.2, "text": "工業園新街某大廈發生事件"},
                    {"start": 7.4, "end": 12.6, "text": "現場消息指送院救治"},
                    {"start": 12.6, "end": 15.1, "text": "傷勢未明"},
                    {"start": 15.3, "end": 18.0, "text": "此外，關於澳門今年統計數據"},
                    {"start": 18.0, "end": 55.0, "text": "後續統計內容"},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            bs, be = auto_detect_broll_window(str(tl), 55.0)
            # Starts right after headline (~2.18s) and ends cleanly right before "此外" (~15.22s)
            self.assertAlmostEqual(bs, 2.18, places=2)
            self.assertAlmostEqual(be, 15.22, places=2)

    def test_cli_defaults_enable_censor_and_image2_alias(self):
        parser = build_parser()
        args = parser.parse_args([
            "--image", "a.jpg",
            "--image2", "b.jpg",
            "--text", "測試文稿",
        ])
        self.assertEqual(args.image, "a.jpg")
        self.assertEqual(args.broll_image, "b.jpg")
        self.assertFalse(args.no_censor)
        self.assertEqual(args.broll_label, "現場資料")
        self.assertFalse(args.fast)


if __name__ == "__main__":
    unittest.main()
