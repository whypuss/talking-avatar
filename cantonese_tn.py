#!/usr/bin/env python3
"""
Cantonese news-broadcast text normalization (TN) & smart sentence/subtitle segmentation.

Separates:
  - display_text: keeps clean editorial numbers & symbols (e.g. 194.9萬人次, 3.4%, 85%, 2026年)
  - tts_text:     expands numbers, decimals, percentages, years, dates, currencies, and ranges
                  into natural Cantonese broadcast Chinese for CosyVoice2-Yue.
"""

from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"

_FULLWIDTH_TRANS = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "％": "%", "＋": "+", "－": "-", "／": "/", "：": ":",
})

# Short transitional openers that should stay attached to the following clause, not trail at the end of a previous segment.
_TRAILING_LEADIN_RE = re.compile(
    r"^(?:明年|今年|去年|同時|此外|另外|然而|不過|因此|例如|至於|"
    r"線上方面|線下方面|社區活動方面|旅遊推廣方面|整體而言|另一方面|"
    r"據了解|據悉|對此|其中|與此同時|換言之)[，,、]?$"
)

# Split subtitles on major punctuation (NOT on '、' ideographic comma, NOT on decimals 194.9 or thousands 1,200).
SUBTITLE_SPLIT_RE = re.compile(r"(?:[，。！？；\n]|(?<!\d)[,\.](?!\d))+")

# Keep atomic numeric/percent/unit tokens intact when hard-wrapping long subtitle lines.
_ATOMIC_TOKEN_RE = re.compile(
    r"(?:(?:HK\$|US\$|MOP|HKD|USD|RMB|CNY)\s*)?[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:%|萬|億|千|百|年|月|日|度|℃|°C|個|名|宗|次)?"
)


def _digits_verbatim(s: str) -> str:
    return "".join(_DIGITS[int(ch)] for ch in s if ch.isdigit())


def int_to_cantonese(n: int, *, two_as_loeng: bool = False) -> str:
    """Convert non-negative integer into standard Cantonese broadcast Chinese reading."""
    if n == 0:
        return "零"
    if n < 0:
        return "負" + int_to_cantonese(-n, two_as_loeng=two_as_loeng)

    def _under_10000(val: int, leading: bool) -> str:
        if val == 0:
            return ""
        units = [("", 1), ("十", 10), ("百", 100), ("千", 1000)]
        parts: list[str] = []
        zero_pending = False
        for idx in range(3, -1, -1):
            unit_char, base = units[idx]
            d = (val // base) % 10
            if d == 0:
                if parts:
                    zero_pending = True
            else:
                if zero_pending:
                    parts.append("零")
                    zero_pending = False
                # Omit leading '一' for 10..19 only when at the start of the whole number
                if d == 1 and base == 10 and leading and not parts:
                    parts.append("十")
                else:
                    parts.append(_DIGITS[d] + unit_char)
        return "".join(parts)

    sections: list[tuple[int, str]] = [
        (10**8, "億"),
        (10**4, "萬"),
        (1, ""),
    ]
    out: list[str] = []
    rem = n
    prev_nonzero = False
    for divisor, sec_unit in sections:
        chunk = rem // divisor
        rem %= divisor
        if chunk == 0:
            continue
        if prev_nonzero and chunk < 1000:
            out.append("零")
        is_leading = not prev_nonzero
        out.append(_under_10000(chunk, leading=is_leading) + sec_unit)
        prev_nonzero = True

    res = "".join(out)
    if two_as_loeng and res == "二":
        return "兩"
    return res


def number_str_to_cantonese(num_str: str) -> str:
    """Convert a decimal or integer numeric string (with optional +/- and commas) into Cantonese reading."""
    s = num_str.strip().replace(",", "")
    if not s:
        return s
    prefix = ""
    if s.startswith("+"):
        prefix = "正"
        s = s[1:]
    elif s.startswith("-"):
        prefix = "負"
        s = s[1:]

    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_read = int_to_cantonese(int(int_part)) if int_part else "零"
        frac_read = _digits_verbatim(frac_part)
        return f"{prefix}{int_read}點{frac_read}"

    # Preserve digit-by-digit reading for leading-zero codes (e.g. 00853) or long IDs/phone numbers (>= 9 digits)
    if (len(s) > 1 and s.startswith("0")) or len(s) >= 9:
        return prefix + _digits_verbatim(s)

    return prefix + int_to_cantonese(int(s))


def normalize_cantonese_tts(text: str) -> str:
    """
    Convert Arabic numbers, decimals, percentages, years, dates, times, and currencies
    into natural Cantonese news-broadcast Chinese while keeping regular Chinese intact.
    """
    if not text:
        return ""

    s = text.translate(_FULLWIDTH_TRANS)
    # Fullwidth decimal dot between digits -> ASCII dot
    s = re.sub(r"(?<=\d)．(?=\d)", ".", s)
    # Strip thousands-separator commas inside numbers (e.g. 1,234,567.8 -> 1234567.8)
    while True:
        new_s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
        if new_s == s:
            break
        s = new_s

    # 1. ISO / slash dates: 2026-11-24 or 2026/11/24
    def _repl_iso_date(m: re.Match) -> str:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{_digits_verbatim(y)}年{int_to_cantonese(mo)}月{int_to_cantonese(d)}日"

    s = re.sub(r"\b((?:19|20)\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", _repl_iso_date, s)

    # 2. 4-digit years followed by 年 (e.g. 2026年, 1999年) -> digit-by-digit 二零二六年
    s = re.sub(
        r"(?<!\d)((?:18|19|20|21)\d{2})(?=年)",
        lambda m: _digits_verbatim(m.group(1)),
        s,
    )

    # 3. Time expressions: 09:30, 14:00, 8:05
    def _repl_time(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{int_to_cantonese(h)}點"
        if mi < 10:
            return f"{int_to_cantonese(h)}點零{int_to_cantonese(mi)}分"
        return f"{int_to_cantonese(h)}點{int_to_cantonese(mi)}分"

    s = re.sub(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", _repl_time, s)

    # 4. Percentage ranges: 3.4%至5.6%, 3%-5%, 3~5%
    def _repl_pct_range(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        return f"百分之{number_str_to_cantonese(a)}至{number_str_to_cantonese(b)}"

    s = re.sub(
        r"([+-]?\d+(?:\.\d+)?)%?\s*(?:至|到|~|—|–|-)\s*([+-]?\d+(?:\.\d+)?)%",
        _repl_pct_range,
        s,
    )

    # 5. Single percentages: +3.4%, -1.2%, 85%, 100%, 0.5%
    def _repl_pct(m: re.Match) -> str:
        raw = m.group(1)
        sign = ""
        if raw.startswith("+"):
            sign = "正"
            raw = raw[1:]
        elif raw.startswith("-"):
            sign = "負"
            raw = raw[1:]
        return f"{sign}百分之{number_str_to_cantonese(raw)}"

    s = re.sub(r"([+-]?\d+(?:\.\d+)?)%", _repl_pct, s)

    # 6. Temperature: 25°C / 25℃ / -3°C
    def _repl_temp(m: re.Match) -> str:
        return f"攝氏{number_str_to_cantonese(m.group(1))}度"

    s = re.sub(r"([+-]?\d+(?:\.\d+)?)\s*(?:°C|℃)", _repl_temp, s)

    # 7. Currencies: MOP 699 / HK$120 / US$50 / RMB 200
    _CURR_MAP = {
        "MOP": "澳門幣",
        "HK$": "港幣",
        "HKD": "港幣",
        "US$": "美元",
        "USD": "美元",
        "RMB": "人民幣",
        "CNY": "人民幣",
    }

    def _repl_currency(m: re.Match) -> str:
        code = m.group(1).upper()
        num = m.group(2)
        trailing = m.group(3) or ""
        label = _CURR_MAP.get(code, code)
        read = number_str_to_cantonese(num)
        if trailing in ("萬", "億", "千", "百", "元", "蚊"):
            return f"{label}{read}{trailing}"
        return f"{label}{read}元{trailing}"

    s = re.sub(
        r"(?<![A-Za-z])(MOP|HKD|USD|RMB|CNY)\s*(\d+(?:\.\d+)?)([萬億千百元蚊]?)",
        _repl_currency,
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"(HK\$|US\$)\s*(\d+(?:\.\d+)?)([萬億千百元蚊]?)",
        _repl_currency,
        s,
        flags=re.IGNORECASE,
    )

    # 8. Ratios & fractions: 3/4 -> 四分之三, 1:2 -> 一比二
    s = re.sub(
        r"(?<!\d)(\d{1,3})/(\d{1,3})(?!\d)",
        lambda m: f"{int_to_cantonese(int(m.group(2)))}分之{int_to_cantonese(int(m.group(1)))}",
        s,
    )
    s = re.sub(
        r"(?<!\d)(\d{1,4}):(\d{1,4})(?!\d)",
        lambda m: f"{int_to_cantonese(int(m.group(1)))}比{int_to_cantonese(int(m.group(2)))}",
        s,
    )

    # 9. Numeric ranges without %: 10-20萬, 3~5日
    s = re.sub(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:~|—|–|-)\s*(\d+(?:\.\d+)?)(?!\d)",
        lambda m: f"{number_str_to_cantonese(m.group(1))}至{number_str_to_cantonese(m.group(2))}",
        s,
    )

    # 10. General decimals and integers (with optional leading +/- when not preceded by digit/CJK-number)
    def _repl_number(m: re.Match) -> str:
        return number_str_to_cantonese(m.group(0))

    s = re.sub(r"(?<![A-Za-z0-9])[+-]?\d+(?:\.\d+)?", _repl_number, s)

    return s


def _speech_len(text: str) -> int:
    """Approximate spoken syllable count using normalized Cantonese text (excluding punctuation)."""
    norm = normalize_cantonese_tts(text)
    cleaned = re.sub(r"[，。！？；、,.\s「」『』（）()“”\"'：:—–\-]", "", norm)
    return max(1, len(cleaned))


def split_tts_segments(text: str, max_speech_len: int = 44, min_speech_len: int = 12) -> list[dict]:
    """
    Split Traditional Chinese broadcast script into natural, balanced TTS segments.

    Returns a list of dicts:
      - display_text: original text for clean subtitles
      - tts_text: Cantonese-normalized text for CosyVoice2-Yue
      - pause_after: 'sentence' (full stop/semicolon) or 'clause' (internal comma split) or 'none'
    """
    clean = re.sub(r"\s+", "", text.strip())
    if not clean:
        return []

    # Split into full sentences by terminal Chinese/ASCII sentence punctuation (never decimal '.')
    raw_sentences = [
        p.strip()
        for p in re.split(r"(?<=[。！？!?；;])", clean)
        if p.strip()
    ]

    # Merge ultra-short standalone sentences into neighbor if combined stays within budget
    merged_sentences: list[str] = []
    for sent in raw_sentences:
        if (
            merged_sentences
            and _speech_len(sent) < min_speech_len
            and _speech_len(merged_sentences[-1]) + _speech_len(sent) <= max_speech_len + 6
        ):
            merged_sentences[-1] += sent
        else:
            merged_sentences.append(sent)

    segments: list[dict] = []

    for sent_idx, sent in enumerate(merged_sentences):
        sent_len = _speech_len(sent)
        if sent_len <= max_speech_len:
            segments.append({
                "display_text": sent,
                "tts_text": normalize_cantonese_tts(sent),
                "pause_after": "sentence" if sent_idx < len(merged_sentences) - 1 else "none",
            })
            continue

        # Split long sentence at commas/ideographic commas (never inside numbers)
        clauses = [
            c
            for c in re.split(r"(?<=[，、])|(?<!\d)(?<=,)(?!\d)", sent)
            if c
        ]
        # Avoid splitting a mildly long sentence (<= max + 10) if splitting would leave an awkward short tail
        if len(clauses) <= 1 or (
            sent_len <= max_speech_len + 10 and min(_speech_len(c) for c in clauses) < min_speech_len
        ):
            segments.append({
                "display_text": sent,
                "tts_text": normalize_cantonese_tts(sent),
                "pause_after": "sentence" if sent_idx < len(merged_sentences) - 1 else "none",
            })
            continue

        # Attach short lead-in clauses (e.g. "線上方面，", "明年，") to the FOLLOWING clause
        grouped_clauses: list[str] = []
        i = 0
        while i < len(clauses):
            c = clauses[i]
            if i + 1 < len(clauses) and (
                _TRAILING_LEADIN_RE.match(c) or _speech_len(c) <= 4
            ):
                clauses[i + 1] = c + clauses[i + 1]
            else:
                grouped_clauses.append(c)
            i += 1
        if not grouped_clauses:
            grouped_clauses = clauses

        # Balanced partition via DP: minimize variance around target length & avoid tiny tails
        n = len(grouped_clauses)
        lens = [_speech_len(c) for c in grouped_clauses]
        total = sum(lens)
        num_parts = max(2, (total + max_speech_len - 1) // max_speech_len)
        target = total / float(num_parts)

        # Prefix sums
        pref = [0] * (n + 1)
        for idx, lv in enumerate(lens):
            pref[idx + 1] = pref[idx] + lv

        # dp[k][j] = min cost to split first j clauses into k parts
        INF = 1e18
        dp = [[INF] * (n + 1) for _ in range(num_parts + 1)]
        parent = [[-1] * (n + 1) for _ in range(num_parts + 1)]
        dp[0][0] = 0.0

        for k in range(1, num_parts + 1):
            for j in range(k, n + 1):
                for prev in range(k - 1, j):
                    if dp[k - 1][prev] >= INF / 2:
                        continue
                    seg_len = pref[j] - pref[prev]
                    # Heavy penalty if way over max or way too short
                    over = max(0, seg_len - max_speech_len)
                    under = max(0, min_speech_len - seg_len)
                    cost = (seg_len - target) ** 2 + (over ** 2) * 12.0 + (under ** 2) * 18.0
                    # Small extra penalty if this split leaves a transitional opener at the end
                    if prev < j and _TRAILING_LEADIN_RE.match(grouped_clauses[j - 1]):
                        cost += 2500.0
                    cand = dp[k - 1][prev] + cost
                    if cand < dp[k][j]:
                        dp[k][j] = cand
                        parent[k][j] = prev

        # Reconstruct partitions (fallback to greedy if n < num_parts)
        parts: list[str] = []
        if n >= num_parts and dp[num_parts][n] < INF / 2:
            cur = n
            for k in range(num_parts, 0, -1):
                prev = parent[k][cur]
                parts.append("".join(grouped_clauses[prev:cur]))
                cur = prev
            parts.reverse()
        else:
            buf = ""
            for c in grouped_clauses:
                if buf and _speech_len(buf) + _speech_len(c) > max_speech_len:
                    parts.append(buf)
                    buf = c
                else:
                    buf += c
            if buf:
                parts.append(buf)

        for p_idx, part in enumerate(parts):
            is_last_part = p_idx == len(parts) - 1
            is_last_sent = sent_idx == len(merged_sentences) - 1
            pause_after = (
                "none" if (is_last_part and is_last_sent)
                else ("sentence" if is_last_part else "clause")
            )
            segments.append({
                "display_text": part,
                "tts_text": normalize_cantonese_tts(part),
                "pause_after": pause_after,
            })

    return segments


def _wrap_line_preserving_numbers(text: str, max_chars: int = 24) -> list[str]:
    """
    Wrap a long subtitle clause cleanly:
      - Never breaks inside a number/decimal/percentage token
      - Prefers breaking after '、' when available
      - Balances multi-line splits so we never leave a 3-char orphan tail like '一千萬'
    """
    if len(text) <= max_chars:
        return [text]

    # Tokenize into atomic numeric units and single characters
    tokens: list[str] = []
    pos = 0
    for m in _ATOMIC_TOKEN_RE.finditer(text):
        if m.start() > pos:
            tokens.extend(list(text[pos : m.start()]))
        tokens.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        tokens.extend(list(text[pos:]))

    total_len = sum(len(t) for t in tokens)
    num_lines = max(2, (total_len + max_chars - 1) // max_chars)
    target = total_len / float(num_lines)

    lines: list[str] = []
    idx = 0
    for line_no in range(num_lines):
        if line_no == num_lines - 1:
            rem = "".join(tokens[idx:])
            if rem:
                lines.append(rem)
            break

        # Choose the best split index near target length without exceeding max_chars
        cur_len = 0
        best_j = idx + 1
        best_score = 1e18
        for j in range(idx + 1, len(tokens)):
            cur_len += len(tokens[j - 1])
            if cur_len > max_chars:
                break
            rem_len = total_len - sum(len(x) for x in lines) - cur_len
            rem_lines = num_lines - line_no - 1
            if rem_len > rem_lines * max_chars + 2:
                continue
            score = abs(cur_len - target)
            # Strong preference to break right after an ideographic comma '、'
            if tokens[j - 1] == "、":
                score -= 4.0
            if score < best_score:
                best_score = score
                best_j = j

        chunk = "".join(tokens[idx:best_j]).rstrip("、")
        if chunk:
            lines.append(chunk)
        idx = best_j

    return lines or [text]


def split_subtitle_lines(text: str, max_chars: int = 24) -> list[str]:
    """
    Split display text into clean, readable Traditional Chinese subtitle lines:
      - NEVER splits inside decimals (194.9, 3.4%) or thousands commas (1,200)
      - Keeps short parallel enumerations with '、' intact unless exceeding max_chars
      - Merges short lead-ins (e.g. '明年', '線上方面') forward, never dangling backward
      - Balances long clauses so no tiny orphan tails remain
    """
    raw_parts = [p.strip() for p in SUBTITLE_SPLIT_RE.split(text) if p.strip()]
    if not raw_parts:
        return []

    merged: list[str] = []
    i = 0
    while i < len(raw_parts):
        cur = raw_parts[i]
        is_leadin = bool(_TRAILING_LEADIN_RE.match(cur))
        if len(cur) <= 5:
            # Prefer merging forward with the next clause
            if i + 1 < len(raw_parts) and len(cur) + 1 + len(raw_parts[i + 1]) <= max_chars + 4:
                raw_parts[i + 1] = f"{cur}，{raw_parts[i + 1]}"
                i += 1
                continue
            # Never merge a transitional lead-in (like '線上方面') backward onto the previous clause
            if not is_leadin and merged and len(merged[-1]) + 1 + len(cur) <= max_chars:
                merged[-1] = f"{merged[-1]}，{cur}"
                i += 1
                continue
        merged.append(cur)
        i += 1

    lines: list[str] = []
    for part in merged:
        lines.extend(_wrap_line_preserving_numbers(part, max_chars=max_chars))
    return lines


def build_synced_subtitle_chunks(
    display_sentence: str,
    start_sec: float,
    end_sec: float,
    max_chars: int = 24,
) -> list[dict]:
    """
    Break a display sentence into readable subtitle lines proportioned by SPOKEN Cantonese syllable count
    (so '3.4%' gets weight for '百分之三點四', staying tightly synced with audio).
    """
    lines = split_subtitle_lines(display_sentence, max_chars=max_chars)
    if not lines:
        return []

    weights = [_speech_len(line) for line in lines]
    total_w = sum(weights)
    dur = max(0.05, end_sec - start_sec)
    cursor = start_sec
    entries: list[dict] = []
    for idx, (line, w) in enumerate(zip(lines, weights)):
        if idx == len(lines) - 1:
            seg_end = end_sec
        else:
            seg_end = cursor + dur * (w / float(total_w))
        entries.append({
            "start": round(cursor, 3),
            "end": round(seg_end, 3),
            "text": line,
        })
        cursor = seg_end
    return entries
