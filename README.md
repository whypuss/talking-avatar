# whytalkv3 (全畫面高清串流 MPS 對口型數字人)

將任意人像圖片與文字（或預先合成的音訊）轉換成**保留完整原始構圖、自然呼吸微動、自然頭部與眨眼動作、高清嘴部與毫秒級同步字幕**的 MP4 講話視頻。專為 **Apple Silicon (M1/M2 8GB 統一記憶體)** 深度最佳化，採用單進程分句 CosyVoice2-Yue + 零硬切串流 Apple MPS Wav2Lip-GAN 架構。

## 一鍵執行 (`whytalkv3.py`)

```bash
cd ~/projects/whytalk && .venv/bin/python whytalkv3.py \
  --image "portrait.jpg" \
  --text "各位觀眾，歡迎收看最新科技快訊。" \
  --voice-ref 女主播h \
  --output output/whytalkv3_out.mp4
```

- 預設啟用 `auto`（Apple MPS GPU）串流推論（`--chunk-sec 0`，徹底消除舊版 6 秒硬切與 8GB OOM）＋ LAB 膚色對齊/嘴周銳化＋保留 24kHz 原聲。
- 加上 `--hd` 可啟用 Apple MPS 加速的 GFPGAN v1.4 高清修復（自動跳過靜音幀）。
- `whytalkv2.py` 保留為向後相容轉發入口，自動呼叫 `whytalkv3.py`。

---

## 模型架構與技術棧總覽

本專案採用多模型串接管線（Pipeline），各階段使用的核心模型與工具如下：

| 階段 | 模型 / 工具 | 用途與特點 | 備註 |
| :--- | :--- | :--- | :--- |
| **1. 語音合成 (TTS)** | **CosyVoice2-Yue** (`jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx`) | 單進程分句粵語零樣本聲線複製；內建 `cantonese_tn.py` 雙軌正規化（發音自動展開數字/小數/百分比/年份，字幕保留原始新聞排版且絕不切斷小數點），輸出 24kHz 原聲與音節加權同步字幕時間軸 | 由 `cosy_batch_tts.py` 載入模型一次完成全部均衡分句合成 |
| | **macOS `say`** (`Sinji` 粵語 / `Tingting` 國語) | 系統內建離線語音合成 | 透過 `--voice Sinji` 直接調用 |
| | **Edge TTS** (`edge-tts`) | 微軟雲端免費神經網路語音 | 透過 `--voice edge:zh-HK-HiuGaaiNeural` 調用 |
| **2. 人臉偵測與關鍵點** | **InsightFace** (`buffalo_l` Lite：`detection` + `landmark_3d_68`) | 僅於原圖執行單次偵測取得臉框、雙眼與 68 點下顎多邊形，後續各幀以精準仿射矩陣追蹤 | 徹底消除逐幀 ONNX 與 S3FD 推論耗時與抖動 |
| **3. 口型同步 (Lip-Sync)** | **Wav2Lip-GAN** (`wav2lip_gan.pth`) | 串流分批（預設 batch=32）於 **Apple MPS / CUDA / CPU** 直接推論，零 6 秒硬切與零 OOM | 預設下巴邊距 `--pads 0 16 8 8` 防左下唇偏移 |
| **4. 畫質修復與融合** | **快速 LAB 色彩對齊 + Unsharp 銳化** / **MPS GFPGAN v1.4** (`--hd`) | 預設以 LAB 膚色對齊 + 嘴周銳化消除 96×96 糊嘴；開啟 `--hd` 時於 MPS 執行 GFPGAN 並自動跳過靜音幀 | 搭配自適應軟羽化 68 點下半臉凸包與 RMS 靜音停頓平滑過渡 |
| **5. 字幕與單次直出** | **快取 RGBA 字幕 + FFmpeg** (`libx264` / `aac 192k`) | 每句字幕僅渲染一次並快取，直接透過 Raw BGR Pipe 單次編碼直出，保留原始 24kHz+ 高清音訊與 `120ms` 口型對齊 | 零中間有損壓縮、支援 `+faststart` 秒開播放 |

---

## 完整執行流程

```text
輸入人像 (Image) ──▶ [1] 全畫面自然呼吸與微動底片生成 (保留原圖背景與上半身構圖)
                              │
輸入文字 / 音訊  ──▶ [2] 語音合成與 16kHz 單聲道 WAV 正規化
                              │
                              ▼
                     [3] Wav2Lip-GAN 時序平滑對口型 (pads: 0 10 0 0)
                              │
                              ▼
                     [4] GFPGAN v1.4 嘴部高清修復
                              │
                              ▼
                     [5] InsightFace 68 點下半臉凸包融合 + RMS 靜音停頓閉嘴過渡
                              │
                              ▼
                     [6] FFmpeg 0.12s 音畫延遲校準與 H.264 MP4 輸出
```

### 流程細節說明
1. **音訊正規化**：無論輸入為文字（TTS）或現成音訊檔（`.wav` / `.mp3` / `.m4a`），皆會自動轉為 **16kHz、單聲道（Mono）、16-bit PCM WAV**，確保 Wav2Lip 梅爾頻譜提取準確。
2. **全畫面自然呼吸與頭部/眼部微動底片**：保留原始圖片完整背景與構圖（自動裁齊偶數寬高以相容 H.264 編碼），並依據物理節奏加入細微的呼吸起伏、自然頭部擺動與低頻眨眼動作（參考 `docs/anchor_motion_spec.md`）。
3. **Wav2Lip-GAN 對口型**：使用 `wav2lip_gan.pth` 進行嘴部動作生成，收緊下巴邊界（`--pads 0 10 0 0`）避免雙下巴偽影。
4. **GFPGAN v1.4 高清修復與凸包融合**：對 Wav2Lip 輸出的模糊嘴部進行超解析度修復，並利用 InsightFace 68 點特徵點構建下半臉凸包遮罩，將修復後的嘴部無縫融合回呼吸底片。
5. **RMS 靜音停頓自動閉嘴**：即時偵測音訊 RMS 能量，當進入靜音停頓區段時平滑過渡回原圖閉嘴表情，消除無聲時的嘴唇抖動。
6. **音畫對齊封裝**：最後透過 FFmpeg 進行 `120ms`（可調）音畫延遲校準，輸出標準 `yuv420p` H.264 MP4。

---

## 快速安裝

系統需求：`Python 3.10–3.12`、`ffmpeg`（macOS 可用 `brew install ffmpeg`）。

```bash
git clone https://github.com/whypuss/talking-avatar.git
cd talking-avatar
bash setup.sh
```

> **重複利用本機既有模型權重**：  
> 若本機已下載 `Wav2Lip/`、`GFPGANv1.4.pth` 或 `insightface` 模型，可透過環境變數直接指定，無須重複下載：
> ```bash
> export WAV2LIP_DIR=/path/to/Wav2Lip
> export GFPGAN_WEIGHTS=/path/to/GFPGANv1.4.pth
> export INSIGHTFACE_ROOT=/path/to/insightface
> ```

---

## 使用方法

```bash
source .venv/bin/activate
```

### 1. 搭配 CosyVoice2-Yue 粵語聲線複製（推薦高品質方案）
先使用 `CosyVoice2-Yue-ZoengJyutGaai-mlx` 根據參考音軌生成粵語 WAV，再傳入 `generate.py`：

```bash
# 步驟 A：生成粵語複製聲線音訊（於 cosyvoice2-yue 環境執行）
~/projects/cosyvoice2-yue/.venv/bin/python3 -m mlx_audio.tts.generate \
  --model jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx \
  --text "各位觀眾大家好，今日為大家報導最新科技新聞。" \
  --ref_audio ref_voice.wav \
  --ref_text "參考音軌的對應逐字稿" \
  --output_path output/cantonese_tts.wav

# 步驟 B：生成對口型高清主播視頻
python generate.py \
  --image anchor.jpg \
  --audio output/cantonese_tts.wav \
  --output output/news_cantonese.mp4
```

> **注意**：凡語音音軌內容、語速或分段時長有變更時，必須重新執行 `generate.py` 生成視頻，切勿直接將新音訊混流至舊版視頻，以免嘴型不同步。

### 2. 圖片 + 內建文字轉語音（macOS `say` / `edge-tts`）
```bash
# 粵語（macOS Sinji）
python generate.py \
  --image your_photo.jpg \
  --text "各位觀眾大家好，今日為大家報導最新科技新聞。" \
  --voice Sinji \
  --output output/news_cantonese.mp4

# 國語（macOS Tingting）
python generate.py \
  --image your_photo.jpg \
  --text "大家好，歡迎收看今天的節目。" \
  --voice Tingting \
  --output output/news_mandarin.mp4

# Edge TTS 粵語女聲
python generate.py \
  --image your_photo.jpg \
  --text "歡迎收看今日新聞快訊。" \
  --voice edge:zh-HK-HiuGaaiNeural \
  --output output/news_edge.mp4
```

### 3. 選用：長文章自動改寫為口播稿 (`summarize.py`)
支援任何 OpenAI 相容 API（透過環境變數設定，切勿將金鑰寫入程式碼）：

```bash
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"

python summarize.py --input article.txt --output script.txt --language Cantonese
python generate.py --image your_photo.jpg --text-file script.txt --output output/broadcast.mp4
```

---

## 完整參數說明

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--image` | **必填** | 任意人像照片（建議正面或微側臉、光線清晰） |
| `--text` / `--text-file` / `--audio` | **三選一** | 要說的文字內容、文字檔路徑，或現成音訊檔（支援 `wav`/`mp3`/`m4a`） |
| `--output` | `output/result.mp4` | 輸出 MP4 檔案路徑 |
| `--voice` | `Sinji` | macOS `say` 聲線（如 `Sinji`、`Tingting`），或 `edge:<voice>` |
| `--rate` | `185` | macOS `say` 語速（每分鐘字數） |
| `--fps` | `25` | 輸出視頻幀率 |
| `--breathing-amp` | `0.6` | 呼吸微動幅度（像素），設為 `0` 可關閉 |
| `--gfpgan-weight` | `0.0` | GFPGAN 嘴部清晰化混合權重（`0` 使用快速 LAB 色彩對齊與銳化；`--hd` 為 `0.65` 並跳過靜音幀） |
| `--pads` | `0 16 8 8` | Wav2Lip 臉部邊界擴展（上 下 左 右） |
| `--audio-delay-ms` | `120` | 音訊延遲毫秒數，對齊中文發音起音 |
| `--device` | `auto` | PyTorch 推論裝置（`auto` 自動優先選擇 `cuda` > Apple `mps` > `cpu`） |
| `--batch-size` | `32` | 串流 Wav2Lip 批次大小（8GB 記憶體安全值，無須 6 秒硬切） |
| `--subtitle-timeline` | `None` | 由 `cosy_batch_tts.py` 輸出的毫秒級精準字幕時間軸 JSON |
| `--wav2lip-dir` | `./Wav2Lip` | Wav2Lip 專案目錄路徑（可由環境變數 `WAV2LIP_DIR` 指定） |
| `--gfpgan-weights` | `./weights/GFPGANv1.4.pth` | GFPGAN 權重檔路徑（可由環境變數 `GFPGAN_WEIGHTS` 指定） |
| `--insightface-root` | `~/.insightface` | InsightFace 模型根目錄（可由環境變數 `INSIGHTFACE_ROOT` 指定） |
| `--keep-temp` | `False` | 保留輸出目錄中的中間暫存檔案以供除錯 |
