# whytalk — AI 執行規範

當使用者上載圖片並要求「用 whytalk 製作視頻」時，請嚴格按照以下流程執行：

1. **準備音訊（若使用者提供文字而非音訊檔）**：
   - **粵語聲線複製（預設優先）**：使用 `~/projects/cosyvoice2-yue/.venv/bin/python3 -m mlx_audio.tts.generate` 搭配模型 `jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx`，必須同時傳入 `--ref_audio` 與 `--ref_text`，產出 WAV 音軌。
   - **或使用內建 TTS**：直接傳入 `--text` 與 `--voice Sinji`（粵語）/ `Tingting`（國語）/ `edge:zh-HK-HiuGaaiNeural`。
2. **生成對口型高清視頻**：
   - 於 `~/projects/whytalk/` 目錄下使用 `.venv/bin/python generate.py`：
     - 傳入使用者上載的圖片 `--image <image_path>`
     - 傳入音訊 `--audio <wav_path>`（或 `--text`）
     - 指定輸出 `--output output/<name>.mp4`
   - 嚴禁直接將新音訊混流至舊視頻，音軌有變更時必須重新執行 `generate.py`。
3. **交付驗證**：
   - 確認輸出 MP4 檔案存在且可正常存取後，回覆使用者檔案路徑或直接發送視頻。
