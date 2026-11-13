# Talking Avatar Pipeline (全畫面高清對口型數字人)

將任意人像圖片與文字（或音訊）轉換成**保留完整原始構圖、自然呼吸微動、高清嘴部與停頓自動閉嘴**的 MP4 講話視頻。專為 **Apple Silicon (M1/M2 8GB 統一記憶體)** 與本地 CPU/GPU 環境最佳化。

## 核心流程

1. **語音合成 / 音訊正規化**：支援 macOS 原生語音（粵語 `Sinji`、國語 `Tingting`）、`edge-tts` 或直接傳入音訊檔，自動轉為 16kHz 單聲道 WAV。
2. **全畫面自然呼吸底片**：保留原始圖片完整背景與上半身構圖，加入細微的呼吸起伏（自動裁齊偶數寬高以相容 H.264）。
3. **Wav2Lip-GAN 時序平滑對口型**：啟用時序平滑並收緊下巴 padding（`0 10 0 0`），讓嘴唇開合貼合語音節奏。
4. **GFPGAN v1.4 高清修復 + 68 點下半臉凸包融合**：修復 Wav2Lip 嘴部像素模糊問題，並偵測音訊 RMS 靜音停頓區段平滑過渡回原圖表情，最後以 `0.12s` 音畫延遲校準輸出。

---

## 快速安裝

需求：`Python 3.10–3.12`、`ffmpeg`（macOS 可用 `brew install ffmpeg`）。

```bash
git clone https://github.com/whypuss/talking-avatar.git
cd talking-avatar
bash setup.sh
```

> 若你本機已經有 `Wav2Lip/`、`weights/GFPGANv1.4.pth` 或 `insightface` 模型，可透過環境變數直接指定，無須重複下載：
> - `WAV2LIP_DIR=/path/to/Wav2Lip`
> - `GFPGAN_WEIGHTS=/path/to/GFPGANv1.4.pth`
> - `INSIGHTFACE_ROOT=/path/to/insightface`

---

## 使用方法（任意圖片皆可）

```bash
source .venv/bin/activate

# 1. 圖片 + 粵語文字（macOS Sinji）
python generate.py \
  --image your_photo.jpg \
  --text "各位觀眾大家好，今日為大家報導最新科技新聞。" \
  --voice Sinji \
  --output output/news_cantonese.mp4

# 2. 圖片 + 國語文字（macOS Tingting）
python generate.py \
  --image your_photo.jpg \
  --text "大家好，歡迎收看今天的節目。" \
  --voice Tingting \
  --output output/news_mandarin.mp4

# 3. 圖片 + 現成音訊檔（支援 wav / mp3 / m4a）
python generate.py \
  --image your_photo.jpg \
  --audio speech.mp3 \
  --output output/from_audio.mp4
```

### 選用：長文章自動改寫為口播稿 (`summarize.py`)

支援任何 OpenAI 相容 API（透過環境變數設定，切勿將金鑰寫入程式碼）：

```bash
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"

python summarize.py --input article.txt --output script.txt --language Cantonese
python generate.py --image your_photo.jpg --text-file script.txt --output output/broadcast.mp4
```

---

## 常用參數說明

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--image` | 必填 | 任意人像照片（建議正面或微側臉、光線清晰） |
| `--text` / `--text-file` / `--audio` | 三選一 | 要說的文字內容或現成音訊檔 |
| `--voice` | `Sinji` | macOS `say` 聲線，或使用 `edge:zh-HK-HiuGaaiNeural` |
| `--breathing-amp` | `0.6` | 呼吸微動幅度（像素），設為 `0` 可關閉 |
| `--gfpgan-weight` | `0.65` | GFPGAN 嘴部清晰化權重（`0` 可跳過以加速） |
| `--pads` | `0 10 0 0` | Wav2Lip 臉部邊界擴展（上 下 左 右） |
| `--audio-delay-ms` | `120` | 音訊延遲毫秒數，對齊中文發音起音 |
| `--device` | `cpu` | GFPGAN 推論裝置（M1 8GB 建議用 `cpu` 防記憶體溢出） |
