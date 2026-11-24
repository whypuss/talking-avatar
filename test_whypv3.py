#!/usr/bin/env python3
"""Unit tests for yt_cover.py and whypv3.py (auto cover copy extraction, safety & CLI defaults)."""

import unittest
from pathlib import Path

from whypv3 import build_parser, derive_cover_path
from yt_cover import extract_cover_copy

QINGZHOU_SCRIPT = (
    "男子青洲疑墮樓送院，工業園新街某大廈中午十二時許發生一宗懷疑墮樓事件。"
    "現場消息指，一名男子懷疑從高處墮下至行人路花圃內，送山頂醫院救治，傷勢未明。"
    "此外，關於澳門今年（2026年）以來涉及墮樓及輕生事件的統計數據，由於特區政府相關部門自去年起"
    "已停止定期對外公開季度自殺死亡監測的詳細分類報告，目前官方並未公佈今年單純「墮樓」類別的具體累計宗數。"
    "不過，根據本地傳媒（澳廣視）向相關渠道獲取的今年上半年整體統計資料顯示，"
    "2026年首六個月澳門合共錄得至少 39宗 自殺身亡個案（當中涵蓋高處墮下等不同方式）；"
    "至於今年下半年至今（截至11月）的官方或傳媒整合總數，則暫未有最新統計公佈。"
)


class TestWhypv3AndCover(unittest.TestCase):
    def test_auto_cover_copy_qingzhou_news(self):
        copy = extract_cover_copy(QINGZHOU_SCRIPT)
        self.assertEqual(copy["badge"], "突發直擊")
        self.assertEqual(copy["location"], "澳門青洲")
        self.assertEqual(copy["line1"], "青洲男子疑墮樓")
        self.assertEqual(copy["line2_prefix"], "半年驚錄")
        self.assertEqual(copy["highlight_val"], "39")
        self.assertEqual(copy["highlight_unit"], "宗")
        self.assertEqual(copy["sub1"], "工業園新街送院")
        self.assertEqual(copy["sub2"], "官方停報季度統計")
        # Strictly zero sensitive terms on cover
        all_text = " ".join(copy.values())
        for bad in ("自殺", "輕生", "死亡", "身亡"):
            self.assertNotIn(bad, all_text)

    def test_auto_cover_copy_tourism_stats(self):
        copy = extract_cover_copy(
            "旅遊局局長文綺華表示，今年首三季訪澳國際旅客約194.9萬人次，按年增長3.4%，估計全年可恢復至疫情前85%水平。"
        )
        self.assertEqual(copy["badge"], "焦點速報")
        self.assertEqual(copy["highlight_val"], "194.9")
        self.assertEqual(copy["highlight_unit"], "萬")
        self.assertEqual(copy["line2_prefix"], "首三季達")

    def test_derive_cover_path_and_cli_defaults(self):
        self.assertEqual(
            derive_cover_path("/tmp/demo_news.mp4"),
            Path("/tmp/demo_news_cover.jpg").resolve(),
        )
        args = build_parser().parse_args([
            "--image", "a.jpg",
            "--image2", "b.jpg",
            "--text", "測試新聞",
        ])
        self.assertFalse(args.no_cover)
        self.assertFalse(args.no_censor)
        self.assertEqual(args.broll_image, "b.jpg")


if __name__ == "__main__":
    unittest.main()
