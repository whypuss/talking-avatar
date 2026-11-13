#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=== 1. Checking system dependencies ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found. Install via 'brew install ffmpeg' (macOS) or apt."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Using Python: $($PYTHON_BIN --version)"

echo "=== 2. Creating virtual environment (.venv) ==="
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 3. Cloning & patching Wav2Lip ==="
if [ ! -d "Wav2Lip" ]; then
  git clone https://github.com/Rudrabha/Wav2Lip.git
fi

# Patch Wav2Lip/audio.py for librosa >= 0.10 compatibility
python - <<'PY'
from pathlib import Path
p = Path("Wav2Lip/audio.py")
text = p.read_text()
old = "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
new = "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"
if old in text:
    p.write_text(text.replace(old, new))
    print("Patched Wav2Lip/audio.py for librosa 0.10+")
PY

# Patch basicsr/data/degradations.py for torchvision >= 0.17 compatibility
python - <<'PY'
import site
from pathlib import Path
for sp in site.getsitepackages():
    p = Path(sp) / "basicsr/data/degradations.py"
    if p.exists():
        text = p.read_text()
        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new = "from torchvision.transforms.functional import rgb_to_grayscale"
        if old in text:
            p.write_text(text.replace(old, new))
            print(f"Patched {p} for torchvision compatibility")
PY

echo "=== 4. Downloading model weights ==="
mkdir -p Wav2Lip/checkpoints Wav2Lip/face_detection/detection/sfd weights

if [ ! -f "Wav2Lip/checkpoints/wav2lip_gan.pth" ]; then
  echo "Downloading wav2lip_gan.pth..."
  curl -L -o Wav2Lip/checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
fi

if [ ! -f "Wav2Lip/face_detection/detection/sfd/s3fd.pth" ]; then
  echo "Downloading s3fd.pth..."
  curl -L -o Wav2Lip/face_detection/detection/sfd/s3fd.pth \
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
fi

if [ ! -f "weights/GFPGANv1.4.pth" ]; then
  echo "Downloading GFPGANv1.4.pth..."
  curl -L -o weights/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
fi

echo "=== Setup complete! ==="
echo "Run: source .venv/bin/activate && python generate.py --image your_photo.jpg --text '你好，歡迎收看今日新聞。' --output output/demo.mp4"
