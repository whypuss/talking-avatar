#!/usr/bin/env python3
"""Unit tests for cantonese_tn.py (number/percent normalization, TTS splitting, subtitle slicing)."""

import unittest
from cantonese_tn import (
    build_synced_subtitle_chunks,
    censor_subtitle_text,
    int_to_cantonese,
    normalize_cantonese_tts,
    split_subtitle_lines,
    split_tts_segments,
)

SAMPLE_SCRIPT = (
    "議員宋碧琪關注政府在開拓國際多元客源方面的實效。"
    "旅遊局局長文綺華回應表示，今年首三季訪澳國際旅客約194.9萬人次，按年增長3.4%。"
    "但中東戰爭推高國際航空燃油價格，旅客出行成本上升，同時澳門直達亞洲地區航班減少，"
    "導致亞洲市場客源出現跌幅，但歐洲等遠程市場客源增長，估計全年國際客量可恢復至疫情前的85%水平。"
    "明年，澳門將鞏固東北亞、東南亞、歐美等傳統市場，同時重點開拓中亞新興市場，"
    "例如把握澳門與哈薩克斯坦免簽政策，以及香港開通前往當地直航航班的機遇，挖掘新增量；"
    "並借助南方航空龐大的國際航線網絡，帶動更多國際旅客訪澳。"
    "局方並會重新審視及優化旅客經香港國際機場禁區至澳門的免費單程直達巴士車票安排，務求觸及更多旅客。"
    "在旅遊推廣上，旅遊局正採取線上線下雙軌並行模式，線上方面，旅遊局運營的二十八個社交賬號合計粉絲總量突破一千萬，"
    "持續產出多語言版本微短劇、微綜藝及短視頻內容，觸達不同國家地區受眾。"
    "線下除舉辦海外路演之外，亦持續與六大綜合休閒企業緊密協作，檢視及監督企業國際客源營銷投放成效，共同做大國際旅客市場。"
    "社區活動方面，文綺華表示，今年國際美食之都嘉年華首次在新口岸區設分會場，明年將延續相關安排，"
    "並持續與新口岸居民、商戶溝通，廣泛收集意見優化活動設計。"
)


class TestCantoneseTN(unittest.TestCase):
    def test_integers(self):
        self.assertEqual(int_to_cantonese(0), "零")
        self.assertEqual(int_to_cantonese(10), "十")
        self.assertEqual(int_to_cantonese(15), "十五")
        self.assertEqual(int_to_cantonese(85), "八十五")
        self.assertEqual(int_to_cantonese(100), "一百")
        self.assertEqual(int_to_cantonese(105), "一百零五")
        self.assertEqual(int_to_cantonese(110), "一百一十")
        self.assertEqual(int_to_cantonese(194), "一百九十四")
        self.assertEqual(int_to_cantonese(10200), "一萬零二百")

    def test_news_numbers_and_percentages(self):
        self.assertEqual(
            normalize_cantonese_tts("今年首三季訪澳國際旅客約194.9萬人次，按年增長3.4%。"),
            "今年首三季訪澳國際旅客約一百九十四點九萬人次，按年增長百分之三點四。",
        )
        self.assertEqual(
            normalize_cantonese_tts("可恢復至疫情前的85%水平。"),
            "可恢復至疫情前的百分之八十五水平。",
        )
        self.assertEqual(
            normalize_cantonese_tts("預計2027年1月1日下調約10%，跌幅-1.5%，介乎3.4%至5.6%。"),
            "預計二零二七年一月一日下調約百分之十，跌幅負百分之一點五，介乎百分之三點四至五點六。",
        )
        self.assertEqual(
            normalize_cantonese_tts("共1,250宗，金額MOP 169.05億，氣溫28.5°C，時間09:05。"),
            "共一千二百五十宗，金額澳門幣一百六十九點零五億，氣溫攝氏二十八點五度，時間九點零五分。",
        )

    def test_subtitle_never_splits_decimals_or_percents(self):
        lines = split_subtitle_lines("旅遊局局長文綺華回應表示，今年首三季訪澳國際旅客約194.9萬人次，按年增長3.4%。")
        joined = "|".join(lines)
        self.assertIn("194.9萬人次", joined)
        self.assertIn("3.4%", joined)
        for line in lines:
            self.assertFalse(line.endswith("194"))
            self.assertFalse(line.startswith("9萬"))
            self.assertFalse(line.endswith("3"))
            self.assertFalse(line.startswith("4%"))

        lines2 = split_subtitle_lines("共1,250名旅客，恢復至疫情前的85%水平。")
        self.assertIn("1,250名旅客", "|".join(lines2))
        self.assertIn("85%", "|".join(lines2))

    def test_tts_segmentation_balance_and_leadins(self):
        segs = split_tts_segments(SAMPLE_SCRIPT)
        self.assertGreaterEqual(len(segs), 10)
        self.assertLessEqual(len(segs), 18)
        for seg in segs:
            # No segment should end with dangling '線上方面，'
            self.assertFalse(seg["display_text"].endswith("線上方面，"), f"Dangling lead-in: {seg}")
            # TTS text must contain zero Arabic digits or %
            self.assertNotRegex(seg["tts_text"], r"[0-9%]", f"Unnormalized TTS text: {seg['tts_text']}")
            # No tiny orphan fragments (< 8 chars)
            self.assertGreaterEqual(len(seg["display_text"]), 10, f"Too short segment: {seg}")

    def test_subtitle_weighting(self):
        chunks = build_synced_subtitle_chunks("今年首三季約194.9萬人次，按年增長3.4%。", 0.0, 10.0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "今年首三季約194.9萬人次")
        self.assertEqual(chunks[1]["text"], "按年增長3.4%")
        # Spoken ratio: 16 vs 11 syllables -> second chunk should get ~40% of duration, not tiny 28%
        dur2 = chunks[1]["end"] - chunks[1]["start"]
        self.assertGreater(dur2, 3.6)

    def test_sensitive_subtitle_censoring_and_parens_tts(self):
        self.assertEqual(
            censor_subtitle_text("涉及墮樓及輕生事件，季度自殺死亡監測，至少39宗自殺身亡個案"),
            "涉及墮樓及xx事件，季度xx監測，至少39宗xx個案",
        )
        tts = normalize_cantonese_tts("今年（2026年）首六個月錄得39宗（當中涵蓋不同方式），截至11月。")
        self.assertIn("今年二零二六年", tts)
        self.assertIn("三十九宗", tts)
        self.assertIn("截至十一月", tts)
        self.assertNotIn("（", tts)
        self.assertNotIn("）", tts)


if __name__ == "__main__":
    unittest.main()
