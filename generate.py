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
    audio_path: str | None = None,
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

    # Detect face once to anchor head-only natural sway/nod while keeping background static
    head_mask = None
    eye_boxes = []
    pivot = (w / 2.0, h / 2.0)
    if amp > 0:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            faces = app.get(img)
            if faces:
                face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
                box = face.bbox
                x1, y1, x2, y2 = [float(v) for v in box]
                fw, fh = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                # Pivot at neck base
                pivot = (cx, min(float(h - 1), y2 + fh * 0.25))
                # Soft elliptical mask covering head + hair
                yy, xx = np.ogrid[:h, :w]
                rx, ry = fw * 0.95, fh * 1.15
                dist = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - (cy - fh * 0.08)) / max(1.0, ry)) ** 2
                mask = np.clip(1.35 - dist, 0.0, 1.0).astype(np.float32)
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(5.0, fw * 0.08))
                head_mask = mask[..., None]
                # Extract left & right eye boxes from 2D landmarks (kps[0]=left_eye, kps[1]=right_eye)
                if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 2:
                    ew = max(12, int(fw * 0.22))
                    eh = max(8, int(fh * 0.11))
                    for ex, ey in face.kps[:2]:
                        ex_i, ey_i = int(round(ex)), int(round(ey))
                        eye_boxes.append((
                            max(0, ex_i - ew // 2),
                            max(0, ey_i - eh // 2),
                            min(w, ex_i + ew // 2),
                            min(h, ey_i + eh // 2),
                        ))
        except Exception:
            head_mask = None

    # Compute per-frame speech energy for natural pauses and accent nods
    rms_env = np.ones(total_frames, dtype=np.float32)
    if audio_path and os.path.exists(audio_path):
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            hop = max(1, int(sr / fps))
            rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
            if len(rms) > 0:
                rms_norm = rms / (np.percentile(rms, 95) + 1e-6)
                rms_norm = np.clip(rms_norm, 0.0, 1.2)
                n = min(total_frames, len(rms_norm))
                rms_env[:n] = rms_norm[:n]
                if n < total_frames:
                    rms_env[n:] = 0.0
        except Exception:
            pass

    # Precompute professional news anchor head trajectory:
    # Upright baseline, subtle accent micro-nods, gentle turn-of-phrase tilt (2-4 deg) & return,
    # and a brief downward glance/nod near ~9s (or ~83% mark for shorter clips).
    dx_arr = np.zeros(total_frames, dtype=np.float32)
    dy_arr = np.zeros(total_frames, dtype=np.float32)
    ang_arr = np.zeros(total_frames, dtype=np.float32)

    cur_dy, cur_ang = 0.0, 0.0
    tgt_ang = 0.0
    tilt_timer = 0
    tilt_dir = 1.0
    nod_timer = 0
    nod_cooldown = 0
    glance_frame = int(min(9.0 * fps, total_frames * 0.83)) if total_frames > int(2.5 * fps) else -1

    for idx in range(total_frames):
        e = float(rms_env[idx])
        is_speaking = e > 0.15

        # Subtle micro-nods synced to vocal emphasis (1-3 deg equivalent)
        if nod_cooldown > 0:
            nod_cooldown -= 1
        if is_speaking and nod_cooldown == 0 and (idx == 0 or rms_env[idx - 1] <= 0.15 or e > 0.70):
            nod_timer = int(fps * 0.42)
            nod_cooldown = int(fps * 0.90)

        # Occasional gentle tilt (2-4 deg) at phrase transitions, then return to 0
        if tilt_timer > 0:
            tilt_timer -= 1
            if tilt_timer == 0:
                tgt_ang = 0.0
        elif is_speaking and idx > 0 and idx % int(fps * 2.8) == 0:
            tgt_ang = tilt_dir * min(3.2, 2.0 * amp)
            tilt_dir *= -1.0
            tilt_timer = int(fps * 1.1)

        # Brief downward nod/glance (~0.6s) around 9s mark (or 83% of short clip)
        glance_dy = 0.0
        if glance_frame > 0 and glance_frame <= idx < glance_frame + int(fps * 0.65):
            gp = (idx - glance_frame) / max(1.0, fps * 0.65)
            glance_dy = 3.5 * amp * math.sin(math.pi * gp)

        alpha = 0.10
        cur_ang += (tgt_ang - cur_ang) * alpha

        nod_dy = 0.0
        if nod_timer > 0:
            p = 1.0 - (nod_timer / max(1.0, fps * 0.42))
            nod_dy = 1.8 * amp * math.sin(math.pi * (p ** 0.65))
            nod_timer -= 1

        cur_dy += ((nod_dy + glance_dy) - cur_dy) * 0.35
        dx_arr[idx] = 0.0
        dy_arr[idx] = cur_dy
        ang_arr[idx] = cur_ang

    # Precompute natural blink schedule (~3x slower: start around 3.0s, then every ~6.5s)
    blink_times = [3.0, 9.5]
    t_next = 16.0
    while t_next < duration + 1.0:
        blink_times.append(t_next)
        t_next += 6.5

    for idx in range(total_frames):
        if amp > 0:
            t = idx / float(fps)
            src_img = img
            if eye_boxes:
                # Compute blink closure factor [0..1] (fast close ~0.07s, fast open ~0.11s)
                blink_amt = 0.0
                for bt in blink_times:
                    dt = t - bt
                    if -0.07 <= dt <= 0.11:
                        blink_amt = 1.0 - abs(dt / (0.07 if dt < 0 else 0.11))
                        blink_amt = max(0.0, min(1.0, blink_amt))
                        break
                # Subtle horizontal micro-saccade for teleprompter reading
                saccade_x = 0.8 * math.sin(2.0 * math.pi * 0.85 * t)
                if blink_amt > 0.02 or abs(saccade_x) > 0.05:
                    src_img = img.copy()
                    for ex1, ey1, ex2, ey2 in eye_boxes:
                        roi = img[ey1:ey2, ex1:ex2]
                        rh, rw = roi.shape[:2]
                        if rh < 4 or rw < 4:
                            continue
                        # Vertical squeeze toward lower eyelid for natural blink + tiny horizontal shift
                        sy = max(0.15, 1.0 - 0.85 * blink_amt)
                        m_eye = np.float32([[1.0, 0.0, saccade_x], [0.0, sy, (1.0 - sy) * rh * 0.65]])
                        warped_eye = cv2.warpAffine(roi, m_eye, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        # Soft elliptical blend mask for eye ROI
                        ey_y, ey_x = np.ogrid[:rh, :rw]
                        emask = np.clip(1.2 - (((ey_x - rw / 2.0) / (rw * 0.48)) ** 2 + ((ey_y - rh / 2.0) / (rh * 0.45)) ** 2), 0.0, 1.0).astype(np.float32)
                        emask = cv2.GaussianBlur(emask, (0, 0), sigmaX=max(1.5, rw * 0.08))[..., None]
                        src_img[ey1:ey2, ex1:ex2] = np.clip(warped_eye * emask + roi * (1.0 - emask), 0, 255).astype(np.uint8)

            scale = 1.0 + 0.0015 * math.sin(2.0 * math.pi * 0.25 * t)
            mat = cv2.getRotationMatrix2D(pivot, float(ang_arr[idx]), scale)
            mat[0, 2] += float(dx_arr[idx])
            mat[1, 2] += float(dy_arr[idx])
            warped = cv2.warpAffine(src_img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if head_mask is not None:
                frame = np.clip(warped.astype(np.float32) * head_mask + img.astype(np.float32) * (1.0 - head_mask), 0, 255).astype(np.uint8)
            else:
                frame = warped
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


def _split_subtitle_segments(text: str, max_chars: int = 16) -> list[str]:
    """Split Traditional Chinese text into readable subtitle segments."""
    import re
    raw_parts = [p.strip() for p in re.split(r"[，。！？；、,\.\n]+", text) if p.strip()]
    segments: list[str] = []
    for part in raw_parts:
        while len(part) > max_chars:
            segments.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            segments.append(part)
    return segments or [text.strip()]


def _load_tc_font(font_size: int):
    """Load a Traditional Chinese font available on macOS/Linux."""
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _build_subtitle_timeline(text: str, total_frames: int) -> list[tuple[int, int, str]]:
    """Allocate frame ranges [start, end) for each subtitle segment by character length."""
    segments = _split_subtitle_segments(text)
    total_chars = sum(max(1, len(s)) for s in segments)
    timeline: list[tuple[int, int, str]] = []
    cursor = 0
    for idx, seg in enumerate(segments):
        if idx == len(segments) - 1:
            end = total_frames
        else:
            span = max(1, round(total_frames * (len(seg) / total_chars)))
            end = min(total_frames, cursor + span)
        timeline.append((cursor, end, seg))
        cursor = end
    return timeline


def _draw_subtitle_frame(frame_bgr: np.ndarray, text: str, font) -> np.ndarray:
    """Draw Traditional Chinese subtitle with semi-transparent backdrop and stroke."""
    from PIL import Image, ImageDraw
    h, w = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x = max(14, int(w * 0.02))
    pad_y = max(8, int(h * 0.012))
    x = (w - tw) // 2
    y = int(h * 0.90) - th

    box = [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y]
    draw.rounded_rectangle(box, radius=10, fill=(0, 0, 0, 155))
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 230),
    )
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


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
    subtitle_text: str | None = None,
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
    roi_box = None
    for i in range(total_frames):
        ret, frame = cap.read()
        ret_base, base_frame = base_cap.read()
        if not ret or not ret_base:
            break

        # Detect face ROI on first frame so GFPGAN & InsightFace only process the face region
        if roi_box is None:
            init_faces = app.get(frame)
            if not init_faces:
                app.prepare(ctx_id=-1, det_size=(320, 320))
                init_faces = app.get(frame)
            if init_faces:
                fx1, fy1, fx2, fy2 = [int(v) for v in init_faces[0].bbox]
                fw, fh = fx2 - fx1, fy2 - fy1
                rx1 = max(0, fx1 - int(fw * 0.6))
                ry1 = max(0, fy1 - int(fh * 0.4))
                rx2 = min(w, fx2 + int(fw * 0.6))
                ry2 = min(h, fy2 + int(fh * 0.6))
                roi_box = (rx1, ry1, rx2, ry2)

        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            crop = frame[ry1:ry2, rx1:rx2]
            base_crop = base_frame[ry1:ry2, rx1:rx2]
        else:
            crop = frame
            base_crop = base_frame

        if restorer is not None:
            _, _, enhanced_crop = restorer.enhance(
                crop,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=gfpgan_weight,
            )
        else:
            enhanced_crop = crop

        faces = app.get(enhanced_crop)
        if faces:
            pts68 = faces[0].landmark_3d_68[:, :2]
            jaw_pts = np.vstack([pts68[3:14], pts68[35:30:-1]])
            last_jaw_poly = cv2.convexHull(jaw_pts.astype(np.int32))

        w_speak = float(speak_weights[i])
        ch, cw = crop.shape[:2]
        if last_jaw_poly is not None:
            mask = np.zeros((ch, cw), dtype=np.float32)
            cv2.fillConvexPoly(mask, last_jaw_poly, 1.0)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask_3c = (mask * w_speak)[:, :, None]
            blended_crop = (enhanced_crop * mask_3c + base_crop * (1.0 - mask_3c)).astype(np.uint8)
        else:
            blended_crop = enhanced_crop

        if roi_box is not None:
            final_frame = base_frame.copy()
            final_frame[ry1:ry2, rx1:rx2] = blended_crop
        else:
            final_frame = blended_crop

        if subtitle_text:
            if i == 0:
                _sub_font = _load_tc_font(max(24, int(final_frame.shape[1] * 0.042)))
                _sub_timeline = _build_subtitle_timeline(subtitle_text, total_frames)
            for _s_start, _s_end, _s_seg in _sub_timeline:
                if _s_start <= i < _s_end:
                    final_frame = _draw_subtitle_frame(final_frame, _s_seg, _sub_font)
                    break

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
    parser.add_argument("--subtitles", action="store_true", help="Burn Traditional Chinese subtitles from --text / --text-file.")
    parser.add_argument("--subtitle-text", help="Explicit Traditional Chinese subtitle text to burn onto the video.")
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
        w, h = build_breathing_base(args.image, base_mp4, duration=duration + 0.5, fps=args.fps, amp=args.breathing_amp, audio_path=audio_wav)
        print(f"Base video: {w}x{h} @ {args.fps}fps")

        print("=== [3/4] Running Wav2Lip-GAN (smoothed) ===")
        run_wav2lip(args.wav2lip_dir, base_mp4, audio_wav, wav2lip_mp4, pads=args.pads)

        sub_txt = args.subtitle_text
        if not sub_txt and args.subtitles:
            if args.text:
                sub_txt = args.text
            elif args.text_file and os.path.exists(args.text_file):
                with open(args.text_file, "r", encoding="utf-8") as f:
                    sub_txt = f.read().strip()

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
            subtitle_text=sub_txt,
        )
        print(f"Done! Output saved to: {args.output}")
    finally:
        if args.keep_temp:
            print(f"Intermediate files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
