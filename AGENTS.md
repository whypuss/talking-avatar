# whytalk — AI 執行規範

當使用者上載圖片並要求「用 whytalk 製作視頻」或「用 whytalkv2 製作視頻」時，請直接執行整合腳本 `whytalkv2.py`：

```bash
cd ~/projects/whytalk && .venv/bin/python whytalkv2.py \
  --image "<圖片路徑>" \
  --text "<繁體中文口播內容>" \
  --voice-ref 女主播h \
  --output output/whytalkv2_out.mp4
```

- `whytalkv2.py` 已整合：CosyVoice2-Yue 粵語聲線複製 + 自動 6 秒分段防 OOM + Wav2Lip-GAN 對口型 + 繁體中文字幕燒錄 + FFmpeg 無損拼接。
- 預設關閉 GFPGAN（`--gfpgan-weight 0`）以確保速度與穩定性；若需開啟高清修復可加 `--hd`。
- 若使用者提供現成音訊檔，改傳 `--audio "<音訊路徑>"`（可搭配 `--subtitle-text "<字幕>"`）。
