#!/usr/bin/env python3
"""
YouTube 1280x720 High-CTR News Cover (Thumbnail) Generator for whypv3.

Composites:
  - Left:  Image 1 (Anchor portrait + channel logo, preserving full shoulders & face)
  - Right: Image 2 (News scene photo, auto white-border trimmed, diagonally split with glowing slash)
           or stylized dark broadcast backdrop if single-image mode
  - Overlay: Auto-extracted (or manually overridden) high-contrast Traditional Chinese headline,
             highlighted numeric hook, breaking badge, and bottom summary strip — strictly free of
             platform-sensitive terms (自殺/輕生/死亡/身亡).

Rendered via Headless Chrome at 2x supersampling (2560x1440 -> 1280x720) for razor-sharp typography.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

# High-risk terms that must NEVER appear on a YouTube thumbnail
_COVER_FORBIDDEN_RE = re.compile(r"(?:自殺身亡|自殺死亡|自殺|輕生|自殘|自盡|尋短|跳樓|上吊|割腕|身亡|死亡)")

_MACAU_LOCATIONS = (
    "青洲", "氹仔", "路環", "新口岸", "祐漢", "黑沙環", "關閘",
    "筷子基", "台山", "下環", "南灣", "西灣", "高士德", "路氹",
    "橫琴", "港珠澳", "澳門",
)


def _sanitize_cover_phrase(text: str) -> str:
    """Remove forbidden YouTube-demonetization terms cleanly without leaving ugly 'xx' on thumbnails."""
    s = _COVER_FORBIDDEN_RE.sub("相關", text)
    s = re.sub(r"相關{2,}", "相關", s)
    return s.strip()


def extract_cover_copy(text: str) -> dict[str, str]:
    """
    Automatically distill a news broadcast script into punchy, YouTube-safe thumbnail copy:
      - badge, location
      - line1 (6-9 chars bold gold headline)
      - line2_prefix, highlight_val, highlight_unit, line2_suffix
      - sub1, sub2 (two crisp summary tags)
    """
    clean = re.sub(r"\s+", "", (text or "").strip())
    if not clean:
        return {
            "badge": "最新焦點",
            "location": "澳門快訊",
            "line1": "澳門最新消息",
            "line2_prefix": "現場直擊",
            "highlight_val": "",
            "highlight_unit": "",
            "line2_suffix": "重點關注！",
            "sub1": "最新現場跟進",
            "sub2": "權威資訊速報",
        }

    # 1. Badge & Location
    is_breaking = any(k in clean for k in ("突發", "墮樓", "墮下", "送院", "火警", "事故", "意外", "救治", "警方", "交通意外"))
    badge = "突發直擊" if is_breaking else "焦點速報"

    loc = "澳門快訊"
    for place in _MACAU_LOCATIONS:
        if place in clean:
            loc = f"澳門{place}" if place != "澳門" else "澳門焦點"
            break

    # 2. Split into clauses
    clauses = [c.strip() for c in re.split(r"[，。！？；;\n]+", clean) if c.strip()]

    # Line 1: crisp punchy core headline (<= 8-9 chars) from opening clause
    first = _sanitize_cover_phrase(clauses[0]) if clauses else "澳門最新消息"
    # Remove trailing generic words if slightly too long
    if len(first) > 9:
        first_short = re.sub(r"^(?:關於|議員|據悉|據了解|本地)", "", first)
        first = first_short[:9] if len(first_short) >= 6 else first[:9]
    elif len(first) == 9 and first.endswith("送院") and "疑墮樓" in first:
        first = first[:-2]  # e.g. "男子青洲疑墮樓送院" -> "男子青洲疑墮樓"
    # Prefer location-first phrasing if "男子青洲..." -> "青洲男子..."
    m_loc_swap = re.match(r"^(男子|女子|老翁|婦人|市民)(青洲|氹仔|路環|新口岸|祐漢|黑沙環)(.+)$", first)
    if m_loc_swap:
        first = f"{m_loc_swap.group(2)}{m_loc_swap.group(1)}{m_loc_swap.group(3)}"
    line1 = first[:9]

    # 3. Highlighted numeric hook (pick the most impactful non-year statistic)
    highlight_val = ""
    highlight_unit = ""
    line2_prefix = ""
    line2_suffix = "！"

    # Search for compelling stats: e.g., 39宗, 194.9萬, 85%, 3.4%, 10%
    stat_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(%|萬人次|萬|億|宗|個|人|倍|元)", clean))
    # Skip bare years or dates like 11月
    chosen = None
    for m in stat_matches:
        val, unit = m.group(1), m.group(2)
        if unit == "萬人次":
            unit = "萬"
        chosen = (val, unit, m.start())
        # Prefer 宗 / % / 萬 / 億
        if unit in ("宗", "%", "萬", "億"):
            break

    if chosen:
        highlight_val, highlight_unit, pos = chosen
        ctx_before = clean[max(0, pos - 18) : pos]
        if any(k in ctx_before for k in ("上半年", "首六個月", "半年")):
            line2_prefix = "半年驚錄" if is_breaking else "半年錄得"
        elif any(k in ctx_before for k in ("首三季", "前三季")):
            line2_prefix = "首三季達"
        elif "增長" in ctx_before:
            line2_prefix = "按年急升"
        elif "下調" in ctx_before or "減" in ctx_before:
            line2_prefix = "最新下調"
        elif "恢復" in ctx_before:
            line2_prefix = "可望恢復"
        elif "突破" in ctx_before:
            line2_prefix = "合計突破"
        else:
            line2_prefix = "最新錄得"
    else:
        # No numeric stat -> punchy second headline from 2nd clause
        second = _sanitize_cover_phrase(clauses[1]) if len(clauses) > 1 else "最新進展曝光"
        line2_prefix = second[:8]
        line2_suffix = "！"

    # 4. Bottom summary strip (two clean non-sensitive highlights)
    sub1 = ""
    sub2 = ""
    if "工業園新街" in clean and "送院" in clean:
        sub1 = "工業園新街送院"
    elif "山頂醫院" in clean:
        sub1 = "送山頂醫院救治"
    elif len(clauses) > 1:
        sub1 = _sanitize_cover_phrase(clauses[1])[:10]
    else:
        sub1 = "現場最新跟進"

    if "停止" in clean and ("統計" in clean or "報告" in clean or "公開" in clean):
        sub2 = "官方停報季度統計"
    elif "暫未有最新統計" in clean:
        sub2 = "下半年總數未公佈"
    elif len(clauses) > 2:
        sub2 = _sanitize_cover_phrase(clauses[-1])[:10]
    else:
        sub2 = "最新資訊詳解"

    return {
        "badge": badge,
        "location": loc,
        "line1": _sanitize_cover_phrase(line1),
        "line2_prefix": _sanitize_cover_phrase(line2_prefix),
        "highlight_val": highlight_val,
        "highlight_unit": highlight_unit,
        "line2_suffix": line2_suffix,
        "sub1": _sanitize_cover_phrase(sub1),
        "sub2": _sanitize_cover_phrase(sub2),
    }


def _trim_white_borders_and_enhance(image_path: str, out_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read scene image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white = gray < 245
    rows = np.where(non_white.mean(axis=1) > 0.20)[0]
    cols = np.where(non_white.mean(axis=0) > 0.20)[0]
    if len(rows) > 20 and len(cols) > 20:
        y1 = min(int(rows[-1]) - 10, int(rows[0]) + 3)
        y2 = max(y1 + 20, int(rows[-1]) - 2)
        x1 = min(int(cols[-1]) - 10, int(cols[0]) + 3)
        x2 = max(x1 + 20, int(cols[-1]) - 2)
        img = img[y1:y2, x1:x2]

    # Subtle clarity boost for thumbnail punch
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return out_path


def _find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    raise FileNotFoundError("Google Chrome / Chromium not found for YouTube cover rendering.")


def _build_cover_html(
    anchor_uri: str,
    scene_uri: str | None,
    copy: dict[str, str],
) -> str:
    badge = html.escape(copy["badge"])
    location = html.escape(copy["location"])
    line1 = html.escape(copy["line1"])
    line2_prefix = html.escape(copy["line2_prefix"])
    h_val = html.escape(copy["highlight_val"])
    h_unit = html.escape(copy["highlight_unit"])
    line2_suffix = html.escape(copy["line2_suffix"])
    sub1 = html.escape(copy["sub1"])
    sub2 = html.escape(copy["sub2"])

    # Auto-fit Line 1 font size so 6-9 CJK chars never overflow the right 630px box
    n1 = max(1, len(copy["line1"]))
    if n1 <= 7:
        line1_px = 80
    elif n1 == 8:
        line1_px = 72
    else:
        line1_px = 64

    # Auto-fit Line 2 font size depending on whether a highlighted number pill is present
    if h_val:
        num_chars = len(copy["line2_prefix"]) + len(copy["highlight_val"]) * 0.75 + len(copy["highlight_unit"]) + 1
        l2_white_px = 68 if num_chars <= 8.5 else (58 if num_chars <= 10.5 else 50)
        num_val_px = 92 if len(copy["highlight_val"]) <= 3 else 74
        num_unit_px = 56 if len(copy["highlight_val"]) <= 3 else 46
        line2_html = (
            f'<span class="line-2-white" style="font-size:{l2_white_px}px">{line2_prefix}</span>'
            f'<span class="num-pop">'
            f'<span class="num-pop-val" style="font-size:{num_val_px}px">{h_val}</span>'
            f'<span class="num-pop-unit" style="font-size:{num_unit_px}px">{h_unit}</span>'
            f'</span>'
            f'<span class="line-2-white" style="font-size:{l2_white_px}px">{line2_suffix}</span>'
        )
    else:
        n2 = max(1, len(copy["line2_prefix"]) + len(copy["line2_suffix"]))
        l2_white_px = 72 if n2 <= 8 else 62
        line2_html = f'<span class="line-2-white" style="font-size:{l2_white_px}px">{line2_prefix}{line2_suffix}</span>'

    sub_total = len(copy["sub1"]) + len(copy["sub2"])
    sub_px = 27 if sub_total <= 16 else (24 if sub_total <= 20 else 21)

    scene_block = (
        f'<img class="scene-img" src="{scene_uri}" />'
        if scene_uri
        else '<div class="scene-fallback"></div>'
    )

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 1280px;
    height: 720px;
    overflow: hidden;
    background: #070b14;
    font-family: "PingFang TC", "Hiragino Sans TC", "Hiragino Sans GB", "Heiti TC", "Noto Sans CJK TC", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .canvas {{
    position: relative;
    width: 1280px;
    height: 720px;
    overflow: hidden;
  }}
  .anchor-bg {{
    position: absolute;
    inset: 0;
    width: 1280px;
    height: 720px;
    object-fit: cover;
    object-position: left center;
    filter: contrast(1.04) saturate(1.06);
  }}
  .scene-wrap {{
    position: absolute;
    inset: 0;
    clip-path: polygon(642px 0, 1280px 0, 1280px 720px, 596px 720px);
  }}
  .scene-img {{
    position: absolute;
    left: 565px;
    top: -185px;
    width: 730px;
    height: 760px;
    object-fit: cover;
    object-position: 68% 92%;
    filter: contrast(1.10) saturate(1.15);
  }}
  .scene-fallback {{
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 78% 28%, #1d3a7a 0%, #0b1630 58%, #050914 100%);
  }}
  .scene-scrim {{
    position: absolute;
    inset: 0;
    background:
      linear-gradient(180deg, rgba(6,12,26,0.34) 0%, rgba(6,12,26,0.06) 24%, rgba(6,12,26,0.14) 44%, rgba(6,10,22,0.84) 64%, rgba(4,7,16,0.97) 100%),
      linear-gradient(90deg, rgba(6,10,22,0.42) 0%, rgba(6,10,22,0.0) 18%);
  }}
  .slash-svg {{
    position: absolute;
    inset: 0;
    width: 1280px;
    height: 720px;
    pointer-events: none;
    z-index: 5;
  }}
  .top-badge-row {{
    position: absolute;
    top: 22px;
    left: 662px;
    right: 26px;
    display: flex;
    align-items: center;
    gap: 12px;
    z-index: 10;
  }}
  .badge-breaking {{
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, #ff1e38 0%, #c8001b 100%);
    color: #ffffff;
    font-size: 29px;
    font-weight: 900;
    letter-spacing: 2px;
    padding: 6px 20px;
    border-radius: 7px;
    border: 2.5px solid rgba(255, 236, 130, 0.95);
    box-shadow: 0 6px 20px rgba(255, 20, 50, 0.58), 0 3px 0 #5c000b;
    transform: skewX(-8deg);
  }}
  .badge-breaking span {{
    display: inline-block;
    transform: skewX(8deg);
  }}
  .badge-loc {{
    background: rgba(8, 14, 28, 0.84);
    color: #ffe84d;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 6px 16px;
    border-radius: 7px;
    border: 2px solid rgba(255, 232, 77, 0.78);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.55);
  }}
  .headline-box {{
    position: absolute;
    left: 622px;
    right: 24px;
    bottom: 44px;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }}
  .line-1 {{
    font-size: {line1_px}px;
    font-weight: 900;
    line-height: 1.06;
    letter-spacing: 0px;
    color: #ffe838;
    background: linear-gradient(180deg, #fff885 0%, #ffd200 56%, #ff9900 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter:
      drop-shadow(0 0 2px #000)
      drop-shadow(3px 0 0 #000)
      drop-shadow(-3px 0 0 #000)
      drop-shadow(0 3px 0 #000)
      drop-shadow(0 -3px 0 #000)
      drop-shadow(3px 3px 0 #000)
      drop-shadow(-3px 3px 0 #000)
      drop-shadow(3px -3px 0 #000)
      drop-shadow(-3px -3px 0 #000)
      drop-shadow(0 9px 14px rgba(0,0,0,0.88));
    white-space: nowrap;
  }}
  .line-2 {{
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-top: 2px;
    white-space: nowrap;
  }}
  .line-2-white {{
    font-weight: 900;
    line-height: 1.05;
    color: #ffffff;
    -webkit-text-stroke: 3.5px #050810;
    paint-order: stroke fill;
    text-shadow:
      0 6px 0 #050810,
      0 10px 20px rgba(0, 0, 0, 0.92);
  }}
  .num-pop {{
    display: inline-flex;
    align-items: baseline;
    background: linear-gradient(135deg, #ff203c 0%, #c9001d 100%);
    color: #fff66b;
    padding: 2px 16px 6px 16px;
    border-radius: 13px;
    border: 4px solid #ffffff;
    box-shadow: 0 8px 24px rgba(255, 20, 50, 0.68), 0 6px 0 #380008;
    transform: skewX(-6deg) translateY(2px);
  }}
  .num-pop-val {{
    font-weight: 900;
    line-height: 1.0;
    font-family: "Arial Black", "Impact", "PingFang TC", sans-serif;
    letter-spacing: -1px;
    text-shadow: 0 5px 0 rgba(0,0,0,0.65);
    transform: skewX(6deg);
    display: inline-block;
  }}
  .num-pop-unit {{
    font-weight: 900;
    color: #ffffff;
    margin-left: 4px;
    text-shadow: 0 4px 0 rgba(0,0,0,0.65);
    transform: skewX(6deg);
    display: inline-block;
  }}
  .sub-strip {{
    margin-top: 10px;
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: rgba(8, 14, 28, 0.90);
    border-left: 7px solid #ff203c;
    border-top: 1.5px solid rgba(255,255,255,0.22);
    border-right: 1.5px solid rgba(255,255,255,0.16);
    border-bottom: 1.5px solid rgba(255,255,255,0.16);
    padding: 9px 18px;
    border-radius: 0 10px 10px 0;
    box-shadow: 0 8px 22px rgba(0,0,0,0.7);
  }}
  .sub-strip .sub-main {{
    font-size: {sub_px}px;
    font-weight: 800;
    color: #f4f7ff;
    letter-spacing: 0.6px;
  }}
  .sub-strip .sub-sep {{
    color: #ff3b52;
    font-size: {max(20, sub_px - 2)}px;
    font-weight: 900;
  }}
  .sub-strip .sub-accent {{
    font-size: {sub_px}px;
    font-weight: 800;
    color: #ffe84d;
    letter-spacing: 0.6px;
  }}
  .outer-glow {{
    position: absolute;
    inset: 0;
    border: 4px solid rgba(255, 255, 255, 0.14);
    pointer-events: none;
    z-index: 20;
  }}
</style>
</head>
<body>
  <div class="canvas">
    <img class="anchor-bg" src="{anchor_uri}" />
    <div class="scene-wrap">
      {scene_block}
      <div class="scene-scrim"></div>
    </div>
    <svg class="slash-svg" viewBox="0 0 1280 720" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="redGlow" x="-20%" y="-10%" width="140%" height="120%">
          <feGaussianBlur stdDeviation="7" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <line x1="642" y1="-10" x2="596" y2="730" stroke="#ff1e38" stroke-width="13" filter="url(#redGlow)" />
      <line x1="642" y1="-10" x2="596" y2="730" stroke="#ffffff" stroke-width="4" />
    </svg>
    <div class="top-badge-row">
      <div class="badge-breaking"><span>{badge}</span></div>
      <div class="badge-loc">{location}</div>
    </div>
    <div class="headline-box">
      <div class="line-1">{line1}</div>
      <div class="line-2">{line2_html}</div>
      <div class="sub-strip">
        <span class="sub-main">{sub1}</span>
        <span class="sub-sep">｜</span>
        <span class="sub-accent">{sub2}</span>
      </div>
    </div>
    <div class="outer-glow"></div>
  </div>
</body>
</html>
"""


def generate_youtube_cover(
    anchor_image: str,
    output_path: str,
    scene_image: str | None = None,
    text: str = "",
    badge: str | None = None,
    location: str | None = None,
    line1: str | None = None,
    line2_prefix: str | None = None,
    highlight_val: str | None = None,
    highlight_unit: str | None = None,
    sub1: str | None = None,
    sub2: str | None = None,
) -> str:
    """Generate a 1280x720 high-CTR YouTube news cover JPEG from anchor + optional scene image + script."""
    anchor_p = Path(anchor_image).resolve()
    if not anchor_p.exists():
        raise FileNotFoundError(f"Anchor image not found: {anchor_p}")

    copy = extract_cover_copy(text)
    if badge is not None:
        copy["badge"] = _sanitize_cover_phrase(badge)
    if location is not None:
        copy["location"] = _sanitize_cover_phrase(location)
    if line1 is not None:
        copy["line1"] = _sanitize_cover_phrase(line1)
    if line2_prefix is not None:
        copy["line2_prefix"] = _sanitize_cover_phrase(line2_prefix)
    if highlight_val is not None:
        copy["highlight_val"] = highlight_val
    if highlight_unit is not None:
        copy["highlight_unit"] = highlight_unit
    if sub1 is not None:
        copy["sub1"] = _sanitize_cover_phrase(sub1)
    if sub2 is not None:
        copy["sub2"] = _sanitize_cover_phrase(sub2)

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    chrome = _find_chrome()

    with tempfile.TemporaryDirectory(prefix="yt_cover_") as tmp:
        tmp_dir = Path(tmp)
        scene_uri = None
        if scene_image and Path(scene_image).exists():
            cleaned = _trim_white_borders_and_enhance(scene_image, str(tmp_dir / "scene_clean.jpg"))
            scene_uri = Path(cleaned).resolve().as_uri()

        html_path = tmp_dir / "cover.html"
        png_2x = tmp_dir / "cover_2x.png"
        html_path.write_text(
            _build_cover_html(anchor_p.as_uri(), scene_uri, copy),
            encoding="utf-8",
        )

        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=1280,720",
                f"--screenshot={png_2x}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        im_2x = cv2.imread(str(png_2x))
        if im_2x is None:
            raise RuntimeError(f"Headless Chrome failed to render screenshot: {png_2x}")
        im_1x = cv2.resize(im_2x, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_p), im_1x, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[YouTube Cover] Saved 1280x720 -> {out_p} | Headline: {copy['line1']} / {copy['line2_prefix']}{copy['highlight_val']}{copy['highlight_unit']}")
    return str(out_p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1280x720 YouTube news cover from anchor + scene image.")
    parser.add_argument("--image", required=True, help="Image 1: Anchor portrait image.")
    parser.add_argument("--image2", "--broll-image", dest="scene_image", help="Image 2: Scene/news photo.")
    parser.add_argument("--text", default="", help="News script used to auto-extract cover headline & stats.")
    parser.add_argument("--output", required=True, help="Output JPEG path (1280x720).")
    parser.add_argument("--badge", help="Override top red badge (e.g. 突發直擊).")
    parser.add_argument("--location", help="Override location pill (e.g. 澳門青洲).")
    parser.add_argument("--line1", help="Override golden main headline.")
    parser.add_argument("--line2-prefix", help="Override second-line prefix text.")
    parser.add_argument("--highlight-val", help="Override highlighted number value (e.g. 39).")
    parser.add_argument("--highlight-unit", help="Override highlighted number unit (e.g. 宗).")
    parser.add_argument("--sub1", help="Override bottom strip left text.")
    parser.add_argument("--sub2", help="Override bottom strip right text.")
    args = parser.parse_args()

    generate_youtube_cover(
        anchor_image=args.image,
        scene_image=args.scene_image,
        text=args.text,
        output_path=args.output,
        badge=args.badge,
        location=args.location,
        line1=args.line1,
        line2_prefix=args.line2_prefix,
        highlight_val=args.highlight_val,
        highlight_unit=args.highlight_unit,
        sub1=args.sub1,
        sub2=args.sub2,
    )


if __name__ == "__main__":
    main()
