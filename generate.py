#!/usr/bin/env python3
"""
End-to-end Image-to-Talking-Avatar pipeline.
Turns any portrait image + text (or audio) into a full-frame HD talking-head MP4:
  1. TTS / audio normalization (16kHz mono WAV)
  2. Subtle full-frame breathing base video
  3. Wav2Lip-GAN with temporal smoothing & tight chin padding
  4. GFPGAN v1.4 HD mouth restoration + 68-pt lower-face convex-hull blending + silence-aware pause closing
  5. FFmpeg H.264 mux with 0.12s Chinese/Cantonese lip-sync alignment
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_audio(
    out_wav: str,
    audio_path: str | None = None,
    text: str | None = None,
    text_file: str | None = None,
    voice: str = "Sinji",
    rate: int = 185,
) -> float:
    """Create 16kHz mono PCM WAV from --audio or TTS (--text / --text-file) and return duration in seconds."""
    if audio_path:
        run_cmd([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav,
        ])
    else:
        if text_file:
            text = Path(text_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Provide either --audio, --text, or --text-file.")

        if voice.startswith("edge:"):
            edge_voice = voice.split(":", 1)[1]
            tmp_mp3 = out_wav + ".mp3"
            run_cmd(["edge-tts", "--voice", edge_voice, "--text", text, "--write-media", tmp_mp3])
            run_cmd([
                "ffmpeg", "-y", "-i", tmp_mp3,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav,
            ])
            os.remove(tmp_mp3)
        else:
            # macOS native `say` (e.g. Sinji for Cantonese, Tingting for Mandarin)
            tmp_aiff = out_wav + ".aiff"
            run_cmd(["say", "-v", voice, "-r", str(rate), "-o", tmp_aiff, text])
            run_cmd([
                "ffmpeg", "-y", "-i", tmp_aiff,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav,
            ])
            os.remove(tmp_aiff)

    data, sr = sf.read(out_wav)
    return len(data) / float(sr)


def build_breathing_base(
    image_path: str,
    out_mp4: str,
    duration: float,
    fps: int = 25,
    amp: float = 0.6,
) -> tuple[int, int]:
    """Generate a full-frame subtle breathing base video with even dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    w_even = w - (w % 2)
    h_even = h - (h % 2)
    if (w_even, h_even) != (w, h):
        img = img[:h_even, :w_even]
        w, h = w_even, h_even

    total_frames = max(1, int(math.ceil(duration * fps)))
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for idx in range(total_frames):
        if amp > 0:
            t = idx / float(fps)
            dy = amp * math.sin(2.0 * math.pi * 0.28 * t)
            scale = 1.0 + 0.0018 * math.sin(2.0 * math.pi * 0.28 * t)
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
            mat[1, 2] += dy
            frame = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        else:
            frame = img
        writer.write(frame)

    writer.release()
    return w, h


def run_wav2lip(
    wav2lip_dir: str,
    base_video: str,
    audio_wav: str,
    out_mp4: str,
    pads: list[int],
) -> None:
    """Run Wav2Lip-GAN with temporal smoothing enabled."""
    checkpoint = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Missing Wav2Lip checkpoint: {checkpoint}")

    cmd = [
        sys.executable,
        os.path.join(wav2lip_dir, "inference.py"),
        "--checkpoint_path", checkpoint,
        "--face", base_video,
        "--audio", audio_wav,
        "--outfile", out_mp4,
        "--pads", *[str(p) for p in pads],
    ]
    run_cmd(cmd, cwd=wav2lip_dir)


def compute_silence_weights(audio_path: str, total_frames: int, fps: int = 25) -> np.ndarray:
    """Compute smooth per-frame speaking weight (0.0 on sustained pauses, 1.0 during speech)."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spf = max(1, int(sr / fps))
    rms = np.array([
        np.sqrt(np.mean(audio[i * spf : min((i + 1) * spf, len(audio))] ** 2))
        if i * spf < len(audio) else 0.0
        for i in range(total_frames)
    ])

    is_silent = rms < 0.012
    weights = np.ones(total_frames, dtype=np.float32)
    run = 0
    for i in range(total_frames):
        if is_silent[i]:
            run += 1
        else:
            if run >= 3:
                weights[i - run : i] = 0.0
            run = 0
    if run >= 3:
        weights[total_frames - run : total_frames] = 0.0

    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    weights = np.convolve(weights, kernel, mode="same")
    return np.clip(weights, 0.0, 1.0)


def restore_and_blend(
    wav2lip_video: str,
    base_video: str,
    audio_wav: str,
    output_mp4: str,
    gfpgan_weights: str,
    insightface_root: str,
    gfpgan_weight: float = 0.65,
    fps: int = 25,
    audio_delay_ms: int = 120,
    device_str: str = "cpu",
) -> None:
    """Restore mouth HD texture with GFPGAN, blend lower-face convex hull, and mux with sync offset."""
    app = FaceAnalysis(name="buffalo_l", root=insightface_root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    restorer = None
    if gfpgan_weight > 0:
        if not os.path.exists(gfpgan_weights):
            raise FileNotFoundError(f"Missing GFPGAN weights: {gfpgan_weights}")
        restorer = GFPGANer(
            model_path=gfpgan_weights,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(device_str),
        )

    cap = cv2.VideoCapture(wav2lip_video)
    base_cap = cv2.VideoCapture(base_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    speak_weights = compute_silence_weights(audio_wav, total_frames, fps)
    temp_video = output_mp4 + ".temp.mp4"
    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_jaw_poly = None
    for i in range(total_frames):
        ret, frame = cap.read()
        ret_base, base_frame = base_cap.read()
        if not ret or not ret_base:
            break

        if restorer is not None:
            _, _, enhanced_talk = restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=gfpgan_weight,
            )
        else:
            enhanced_talk = frame

        faces = app.get(enhanced_talk)
        if not faces and i == 0:
            app.prepare(ctx_id=-1, det_size=(320, 320))
            faces = app.get(enhanced_talk)

        if faces:
            pts68 = faces[0].landmark_3d_68[:, :2]
            jaw_pts = np.vstack([pts68[3:14], pts68[35:30:-1]])
            last_jaw_poly = cv2.convexHull(jaw_pts.astype(np.int32))

        w_speak = float(speak_weights[i])
        if last_jaw_poly is not None:
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillConvexPoly(mask, last_jaw_poly, 1.0)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask_3c = (mask * w_speak)[:, :, None]
            final_frame = (enhanced_talk * mask_3c + base_frame * (1.0 - mask_3c)).astype(np.uint8)
        else:
            final_frame = enhanced_talk

        writer.write(final_frame)
        if (i + 1) % 25 == 0 or (i + 1) == total_frames:
            print(f"[HD Restore] {i + 1}/{total_frames} frames processed")

    cap.release()
    base_cap.release()
    writer.release()

    delay_sec = audio_delay_ms / 1000.0
    filter_complex = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={delay_sec:.2f}[v];"
        f"[1:a]adelay={audio_delay_ms}|{audio_delay_ms}[a]"
    )
    run_cmd([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_wav,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_mp4,
    ])
    if os.path.exists(temp_video):
        os.remove(temp_video)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    default_insightface = os.environ.get(
        "INSIGHTFACE_ROOT",
        "/Volumes/Lexar/ComfyUI/models/insightface"
        if Path("/Volumes/Lexar/ComfyUI/models/insightface").exists()
        else str(Path.home() / ".insightface"),
    )

    parser = argparse.ArgumentParser(description="Generate HD talking-avatar video from any portrait image.")
    parser.add_argument("--image", required=True, help="Input portrait image path.")
    parser.add_argument("--text", help="Text to speak (uses TTS).")
    parser.add_argument("--text-file", help="Text file to speak (uses TTS).")
    parser.add_argument("--audio", help="Existing audio file (skips TTS).")
    parser.add_argument("--output", default="output/result.mp4", help="Output MP4 path.")
    parser.add_argument("--voice", default="Sinji", help="macOS `say` voice (e.g. Sinji, Tingting) or `edge:<voice>`.")
    parser.add_argument("--rate", type=int, default=185, help="Speech rate (words per minute) for macOS `say`.")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS.")
    parser.add_argument("--breathing-amp", type=float, default=0.6, help="Subtle breathing amplitude in pixels (0 to disable).")
    parser.add_argument("--gfpgan-weight", type=float, default=0.65, help="GFPGAN restoration blend weight (0 to skip).")
    parser.add_argument("--pads", type=int, nargs=4, default=[0, 10, 0, 0], help="Wav2Lip face padding: top bottom left right.")
    parser.add_argument("--audio-delay-ms", type=int, default=120, help="Audio delay in ms for lip-sync alignment.")
    parser.add_argument("--device", default="cpu", help="PyTorch device for GFPGAN (cpu or cuda).")
    parser.add_argument("--wav2lip-dir", default=os.environ.get("WAV2LIP_DIR", str(repo_dir / "Wav2Lip")))
    parser.add_argument("--gfpgan-weights", default=os.environ.get("GFPGAN_WEIGHTS", str(repo_dir / "weights" / "GFPGANv1.4.pth")))
    parser.add_argument("--insightface-root", default=default_insightface)
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files in output directory.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="talking_avatar_")

    try:
        audio_wav = os.path.join(work_dir, "audio_16k.wav")
        base_mp4 = os.path.join(work_dir, "breathing_base.mp4")
        wav2lip_mp4 = os.path.join(work_dir, "wav2lip_smooth.mp4")

        print("=== [1/4] Preparing audio ===")
        duration = prepare_audio(
            audio_wav,
            audio_path=args.audio,
            text=args.text,
            text_file=args.text_file,
            voice=args.voice,
            rate=args.rate,
        )
        print(f"Audio duration: {duration:.2f}s")

        print("=== [2/4] Building full-frame breathing base ===")
        w, h = build_breathing_base(args.image, base_mp4, duration=duration + 0.5, fps=args.fps, amp=args.breathing_amp)
        print(f"Base video: {w}x{h} @ {args.fps}fps")

        print("=== [3/4] Running Wav2Lip-GAN (smoothed) ===")
        run_wav2lip(args.wav2lip_dir, base_mp4, audio_wav, wav2lip_mp4, pads=args.pads)

        print("=== [4/4] GFPGAN HD restoration & silence-aware blending ===")
        restore_and_blend(
            wav2lip_video=wav2lip_mp4,
            base_video=base_mp4,
            audio_wav=audio_wav,
            output_mp4=args.output,
            gfpgan_weights=args.gfpgan_weights,
            insightface_root=args.insightface_root,
            gfpgan_weight=args.gfpgan_weight,
            fps=args.fps,
            audio_delay_ms=args.audio_delay_ms,
            device_str=args.device,
        )
        print(f"Done! Output saved to: {args.output}")
    finally:
        if args.keep_temp:
            print(f"Intermediate files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
