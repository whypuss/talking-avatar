# whytalkv3 — AI 執行規範

當使用者上載圖片並要求「用 whytalk 製作視頻」、「用 whytalkv3 製作視頻」（或舊稱 `whytalkv2`）時，請直接執行整合腳本 `whytalkv3.py`：

```bash
cd ~/projects/whytalk && .venv/bin/python whytalkv3.py \
  --image "<圖片路徑>" \
  --text "<繁體中文口播內容>" \
  --voice-ref 女主播h \
  --output output/whytalkv3_out.mp4
```

- `whytalkv3.py` 已整合：單進程分句 CosyVoice2-Yue 粵語聲線複製（4.7s 乾淨句尾參考音 + 防殘音補靜音 + 24kHz 原聲保留）+ 零硬切串流 Apple MPS Wav2Lip-GAN（1-shot InsightFace Lite 仿射追蹤，徹底消除 6 秒跳切與 8GB OOM）+ LAB 膚色對齊/嘴周銳化 + 毫秒級同步繁體中文字幕燒錄 + FFmpeg 單次直出。
- 預設使用快速 LAB 色彩對齊 + 局部銳化（`--gfpgan-weight 0`）；若需開啟 MPS 加速 GFPGAN v1.4 高清修復（自動跳過靜音幀）可加 `--hd`。
- 若使用者提供現成音訊檔，改傳 `--audio "<音訊路徑>"`（可搭配 `--subtitle-text "<字幕>"` 或 `--subtitle-timeline "<timeline.json>"`）。
- 舊路徑 `whytalkv2.py` 保留為相容轉發入口，自動轉交 `whytalkv3.py` 執行。
