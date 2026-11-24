#!/usr/bin/env python3
"""
Text-driven Cantonese Viseme (lip shape) modulation engine for whytalkv3.

Maps Traditional Chinese broadcast scripts + subtitle timelines (with Cantonese TN expansion)
into smooth per-frame mouth-shape trajectories:
  - open_scale  : vertical jaw/lip openness (larger on open vowels /aa,a/, pinched on bilabial onset /b,p,m/)
  - width_scale : horizontal lip rounding vs spreading (<1.0 rounded /o,u,yu,oe/, >1.0 spread /i,e,ei/)

Applied via a smooth Gaussian-weighted local displacement field strictly inside the mouth region
so surrounding skin and cheeks stay 100% distortion-free.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

try:
    from cantonese_tn import normalize_cantonese_tts
except Exception:
    def normalize_cantonese_tts(text: str) -> str:  # type: ignore[misc]
        return text


# Cantonese bilabial-onset characters (b / p / m / f-labiodental) -> crisp lip closure at syllable onset
_BILABIAL_CHARS = set(
    "八百半班般搬板版辦伴扮巴把霸爸包保報寶暴爆悲碑卑北奔本崩比彼必畢筆壁碧被備避秘邊編遍便變表標票飄"
    "兵並病冰餅波播博薄步部布佈捕補不培陪賠配佩批披疲脾皮片偏篇平評憑屏品聘貧頻破頗普譜鋪撲僕"
    "馬碼媽麻嗎買賣埋滿慢漫蠻曼忙盲茫毛貿矛冒帽梅媒美每妹味未尾微維危米迷秘秘密蜜免面棉眠民敏名明鳴命"
    "模磨魔摩某謀母木目牧幕墓募萬灣彎玩頑晚挽網忘望旺亡王文聞問蚊物勿務武舞五午無巫"
    "方芳訪防房放法罰乏發飛非肥費廢肺分紛粉份奮憤風豐封峰鋒逢縫否佛福服復複覆幅副富付負婦附赴"
)

# Cantonese rounded-vowel characters (o / u / yu / oe / eo / ou / ui / un / ung / ok / uk) -> gentle lip rounding
_ROUNDED_CHARS = set(
    "澳奧歐偶嘔區驅居車舉據巨聚去趣女旅呂慮律於魚餘余愚宇雨語預遇御育玉獄欲浴域如儒乳入"
    "局足族促續俗屬束叔熟縮祝竹築逐燭曲屈出術述輸書舒鼠暑數樹素訴塑蘇甦租組祖阻做坐座左佐鎖所"
    "都度渡到道導島倒禱刀討套逃陶桃早造操曹草高告稿好號豪耗老勞撈腦惱報保寶暴毛貿澳"
    "中忠終種重仲眾蟲充衝崇通同童銅統痛動洞東冬凍懂公工功攻紅洪宏鴻空控恐龍隆農濃容融榮用擁勇永總宗綜從松宋送頌"
    "國果過郭廣光港講崗剛鋼康抗網往旺放訪房防方旁龐當黨堂唐糖湯浪忙幫邦商賞上尚常場長廠唱張章障漲創窗床雙爽黃皇王"
    "落樂學確作各錯擴活或獲貨火夥和荷何河賀可歌哥個多拖羅螺波播博薄莫膜幕摸磨魔"
    "會回灰恢水誰睡稅瑞推腿退追隊對內外海開該改蓋愛礙賽菜彩在載再待代袋貸太態台來耐"
    "口後候厚收手首守受售壽投頭透偷走奏酒就舊救究久九有友又由郵油遊優憂猶尤右幼誘"
    "路露魯爐蘆旅預序聚續需須序敘徐許虛墟居據舉去區"
)

# Cantonese spread-vowel characters (i / e / ei / in / ing / ip / it / ik) -> gentle lip corners spread
_SPREAD_CHARS = set(
    "一乙億益易譯亦意意義議醫依衣移宜疑以已異利理里離力歷立列烈連臉練聯廉"
    "地第低底抵的滴敵迪體替提題梯鐵貼必畢筆壁碧比彼被備避秘皮批披疲脾片遍便變邊編扁辨辯米迷秘密蜜滅免面"
    "西細世勢計繼季際濟齊制製滯智致置治持遲池時是事示視試思司私師斯施詩資滋支知之指止紙旨子紫自字次此四死"
    "起企期奇其旗器氣棄機基積績激極即級及急吸給業協接節切設舌熱日"
    "年連聯電點添天先線現見建件健千前錢淺全權展戰政正整證成城承程呈稱升聲星醒性姓興行型形經京景警境競敬"
    "定訂停庭聽清青輕情晴請領零令另英應營迎影永平評憑屏兵並病明名精晶淨靜微維為位威唯未味尾美民敏"
    "議員琪綺季際市企微短劇視頻"
)

# Cantonese wide-open vowel characters (aa / a / ai / au / am / an / ang / ak / at) -> fuller vertical opening
_OPEN_CHARS = set(
    "大家發訪增化話加假價架駕嘉佳下夏廈雅瓦花華畫劃跨誇查茶差察沙紗殺打他它她拿那拉啦馬碼媽麻嗎"
    "巴把霸爸怕爬帕法罰乏答達塔踏納臘辣雜閘插刷撒薩哈卡客格革額白百拍伯柏拆策冊責宅窄擇澤獲核嚇黑克刻"
    "得德特勒測側色塞北國廣講港行航剛崗鋼康抗昂當黨檔堂唐糖湯燙浪朗郎忙盲茫幫邦綁旁龐房防放芳方仿"
    "商傷賞上尚常場長廠唱暢張章障漲掌丈帳創窗床雙爽光狂況礦旺往網望忘王黃皇荒慌相香鄉響享向象像項降強搶牆將獎"
    "央陽洋揚楊養樣讓良量涼梁糧亮兩輛藍覽南男難但單擔彈淡蛋談痰坦嘆炭探產慘餐殘燦參三山散刪反返飯販範犯番翻煩繁"
    "萬晚灣彎玩頑挽環還換患幻班般搬板版辦半伴扮盼判攀盤慢漫蠻滿曼間簡減揀監鑑艦站暫戰讚贊棧衫佔斬鹹喊涵含函寒韓漢"
    "限汗旱看刊勘砍幹桿趕感敢甘柑安按案暗岸眼顏研嚴驗演延沿言燕宴厭鹽炎染掩"
)

_PUNCT_RE = re.compile(r"[，。！？；、,.\s「」『』（）()“”\"'：:—–\-]")


def classify_char_viseme(ch: str) -> tuple[float, float, bool]:
    """Return (open_scale, width_scale, is_bilabial) for a single spoken Cantonese character."""
    is_bilab = ch in _BILABIAL_CHARS
    if ch in _ROUNDED_CHARS:
        open_s, width_s = 1.03, 0.945
    elif ch in _SPREAD_CHARS:
        open_s, width_s = 0.96, 1.045
    elif ch in _OPEN_CHARS:
        open_s, width_s = 1.09, 1.01
    else:
        open_s, width_s = 1.01, 1.00
    return open_s, width_s, is_bilab


def _gaussian_smooth_1d(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    if len(arr) < 3 or sigma <= 0:
        return arr
    radius = max(1, int(round(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= float(k.sum())
    padded = np.pad(arr, radius, mode="edge")
    return np.convolve(padded, k, mode="valid").astype(np.float32)


def build_viseme_trajectory(
    total_frames: int,
    fps: int,
    speak_weights: np.ndarray,
    subtitle_timeline: str | None = None,
    subtitle_text: str | None = None,
    audio_delay_ms: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build smooth per-frame (open_scale, width_scale) arrays driven by Cantonese text + timing.
    Values default to 1.0 during silence and modulate gently (~0.92..1.10) during active speech.
    """
    open_arr = np.ones(total_frames, dtype=np.float32)
    width_arr = np.ones(total_frames, dtype=np.float32)

    entries: list[tuple[float, float, str]] = []
    if subtitle_timeline and os.path.exists(subtitle_timeline):
        try:
            raw = json.loads(Path(subtitle_timeline).read_text(encoding="utf-8"))
            for item in raw:
                txt = str(item.get("text", "")).strip()
                if txt:
                    entries.append((float(item["start"]), float(item["end"]), txt))
        except Exception:
            entries = []
    elif subtitle_text:
        entries.append((0.0, total_frames / float(max(1, fps)), subtitle_text.strip()))

    if not entries:
        return open_arr, width_arr

    # Notice: video frames lead audio by audio_delay_ms during muxing, so frame t corresponds to audio time t
    for start_s, end_s, disp_txt in entries:
        spoken = _PUNCT_RE.sub("", normalize_cantonese_tts(disp_txt))
        if not spoken:
            continue
        span = max(0.05, end_s - start_s)
        n_ch = len(spoken)
        char_dur = span / float(n_ch)

        for idx_ch, ch in enumerate(spoken):
            t0 = start_s + idx_ch * char_dur
            t1 = t0 + char_dur
            f0 = max(0, int(round(t0 * fps)))
            f1 = min(total_frames, int(round(t1 * fps)))
            if f1 <= f0:
                continue
            o_s, w_s, is_bilab = classify_char_viseme(ch)
            open_arr[f0:f1] = o_s
            width_arr[f0:f1] = w_s
            # Subtle bilabial closure pinch on the very first 25% of syllable onset
            if is_bilab and (f1 - f0) >= 2:
                pinch_end = max(f0 + 1, f0 + int(round((f1 - f0) * 0.28)))
                open_arr[f0:pinch_end] = 0.87

    # Gate strictly by smooth speaking envelope so silent frames remain 100% neutral (1.0, 1.0)
    sw = speak_weights[:total_frames] if len(speak_weights) >= total_frames else np.ones(total_frames, np.float32)
    open_arr = 1.0 + (open_arr - 1.0) * sw
    width_arr = 1.0 + (width_arr - 1.0) * sw

    # Smooth over ~60ms (sigma=1.5 frames at 25fps) for organic muscle continuity
    open_arr = _gaussian_smooth_1d(open_arr, sigma=1.5)
    width_arr = _gaussian_smooth_1d(width_arr, sigma=1.6)
    return open_arr, width_arr


def apply_viseme_mouth_warp(
    patch_bgr: np.ndarray,
    mouth_center: tuple[float, float],
    mouth_w: float,
    mouth_h: float,
    open_scale: float,
    width_scale: float,
) -> np.ndarray:
    """
    Apply an ultra-smooth Gaussian-radial local warp strictly inside the mouth zone
    to reflect Cantonese text visemes (rounded / spread / open / bilabial closure).
    """
    if abs(open_scale - 1.0) < 0.008 and abs(width_scale - 1.0) < 0.008:
        return patch_bgr

    ph, pw = patch_bgr.shape[:2]
    if ph < 16 or pw < 16:
        return patch_bgr

    cx, cy = float(mouth_center[0]), float(mouth_center[1])
    rx = max(10.0, float(mouth_w) * 0.72)
    ry = max(8.0, float(mouth_h) * 1.15)

    # Local bounding box to avoid remapping the entire face patch
    x1 = max(0, int(cx - rx * 2.2))
    x2 = min(pw, int(cx + rx * 2.2))
    y1 = max(0, int(cy - ry * 2.2))
    y2 = min(ph, int(cy + ry * 2.2))
    rw, rh = x2 - x1, y2 - y1
    if rw < 12 or rh < 12:
        return patch_bgr

    roi = patch_bgr[y1:y2, x1:x2]
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    dx = xx - (cx - x1)
    dy = yy - (cy - y1)

    # Gaussian falloff ensures displacement goes smoothly to 0 at boundary (zero seams)
    r2 = (dx / rx) ** 2 + (dy / ry) ** 2
    falloff = np.exp(-0.55 * r2).astype(np.float32)

    # Inverse mapping: to scale feature by s, sample from offset / s => shift = offset * (1/s - 1)
    inv_w = (1.0 / np.clip(width_scale, 0.88, 1.12)) - 1.0
    inv_o = (1.0 / np.clip(open_scale, 0.84, 1.16)) - 1.0

    map_x = (xx + dx * inv_w * falloff).astype(np.float32)
    map_y = (yy + dy * inv_o * falloff).astype(np.float32)

    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = patch_bgr.copy()
    out[y1:y2, x1:x2] = warped
    return out
