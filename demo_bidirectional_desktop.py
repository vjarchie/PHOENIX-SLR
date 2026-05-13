# -*- coding: utf-8 -*-
"""
Desktop demo: bidirectional sign ↔ text (prototype).

Mode A — Sign → text: webcam buffer + hybrid SLR checkpoint (same stack as demo_camera).
Mode B — Text → sign: German text → rule-based glosses → PHOENIX clip retrieval
         (same stack as demo_speech_to_sign).

Run from project root:
    python demo_bidirectional_desktop.py
    python demo_bidirectional_desktop.py --checkpoint checkpoints/e2e/best.pth --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_camera import SignLanguageDemo


def load_speech_to_sign(root: Path):
    """Load SpeechToSignPipeline + retriever index (mirrors demo_speech_to_sign)."""
    from src.speech_to_sign.pipeline import SpeechToSignPipeline

    data_dir = root / "data" / "phoenix2014-release"
    if not data_dir.exists():
        return None, "PHOENIX data not found at data/phoenix2014-release"

    vocab_path = root / "checkpoints" / "hybrid" / "vocab.json"
    gloss_vocab = {}
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            gloss_vocab = json.load(f)

    index_path = root / "gloss_video_index.pkl"
    pipeline = SpeechToSignPipeline.from_rule_based(str(root), gloss_vocab)

    corpus = (
        data_dir
        / "phoenix-2014-multisigner"
        / "annotations"
        / "manual"
        / "train.corpus.csv"
    )
    if index_path.exists():
        pipeline.gloss_retriever.load_index(str(index_path))
    elif corpus.exists():
        pipeline.gloss_retriever.build_index(str(corpus), save_path=str(index_path))
    else:
        return None, f"Need {index_path} or corpus at {corpus}"

    return pipeline, None


class BidirectionalDemoApp:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        project_root: Path,
        camera_id: int,
    ):
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext

        self.tk = tk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("PHOENIX-SLR — Bidirectional demo")
        self.root.minsize(720, 560)

        self.project_root = project_root
        self.camera_id = camera_id
        self._buffer_lock = threading.Lock()
        self._cam_stop = threading.Event()
        self._cam_thread: Optional[threading.Thread] = None
        self._latest_photo: Optional[ImageTk.PhotoImage] = None
        self._pipeline: Any = None
        self._pipeline_err: Optional[str] = None
        self._sign_video_frames: Optional[np.ndarray] = None
        self._sign_video_idx = 0
        self._after_play_id: Optional[str] = None

        self.demo = SignLanguageDemo(str(checkpoint), device)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Tab 1: Sign → text ---
        tab1 = ttk.Frame(nb, padding=6)
        nb.add(tab1, text="Sign → text (camera)")

        cam_row = ttk.Frame(tab1)
        cam_row.pack(fill=tk.X)
        self.btn_cam = ttk.Button(cam_row, text="Start camera", command=self._toggle_camera)
        self.btn_cam.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(cam_row, text="Clear buffer", command=self._clear_buffer).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(cam_row, text="Decode now", command=self._decode_sign).pack(
            side=tk.LEFT, padx=4
        )
        self.lbl_cam_status = ttk.Label(cam_row, text="Camera off")
        self.lbl_cam_status.pack(side=tk.LEFT, padx=12)

        self.lbl_preview = ttk.Label(tab1)
        self.lbl_preview.pack(pady=6)

        ttk.Label(tab1, text="Glosses (DGS):").pack(anchor=tk.W)
        self.txt_gloss = scrolledtext.ScrolledText(tab1, height=3, wrap=tk.WORD)
        self.txt_gloss.pack(fill=tk.BOTH, expand=False)

        ttk.Label(tab1, text="English (phrase table):").pack(anchor=tk.W, pady=(8, 0))
        self.txt_en = scrolledtext.ScrolledText(tab1, height=3, wrap=tk.WORD)
        self.txt_en.pack(fill=tk.BOTH, expand=True)

        # --- Tab 2: Text → sign ---
        tab2 = ttk.Frame(nb, padding=6)
        nb.add(tab2, text="Text → sign (video)")

        ttk.Label(tab2, text="German sentence (demo / rule-based glosses):").pack(anchor=tk.W)
        self.entry_de = ttk.Entry(tab2, width=80)
        self.entry_de.pack(fill=tk.X, pady=4)
        self.entry_de.insert(
            0, "Morgen gibt es Regen im Norden"
        )

        row2 = ttk.Frame(tab2)
        row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="Load text→sign pipeline", command=self._load_pipeline_async).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.lbl_pipe = ttk.Label(row2, text="Not loaded")
        self.lbl_pipe.pack(side=tk.LEFT)

        ttk.Button(tab2, text="Generate & play", command=self._generate_sign_video).pack(
            anchor=tk.W, pady=4
        )

        self.lbl_sign = ttk.Label(tab2)
        self.lbl_sign.pack(pady=8)

        ttk.Label(tab2, text="Glosses used:").pack(anchor=tk.W)
        self.txt_gloss_out = scrolledtext.ScrolledText(tab2, height=4, wrap=tk.WORD)
        self.txt_gloss_out.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_camera(self) -> None:
        if self._cam_thread is not None:
            self._cam_stop.set()
            self._cam_thread.join(timeout=2.0)
            self._cam_thread = None
            self._cam_stop.clear()
            self.btn_cam.configure(text="Start camera")
            self.lbl_cam_status.configure(text="Camera off")
            return

        self._cam_stop.clear()

        def loop() -> None:
            cap = None
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(self.camera_id, backend)
                if cap.isOpened():
                    ret, fr = cap.read()
                    if ret and fr is not None:
                        break
                if cap:
                    cap.release()
                cap = None
            if cap is None:
                self.root.after(
                    0,
                    lambda: self.messagebox.showerror(
                        "Camera", "Could not open webcam."
                    ),
                )
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            while not self._cam_stop.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                proc = self.demo.preprocess_frame(frame)
                with self._buffer_lock:
                    self.demo.frame_buffer.append(proc)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (480, 360))
                im = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image=im)
                self._latest_photo = photo
                self.root.after(0, lambda p=photo: self.lbl_preview.configure(image=p))

            cap.release()

        self._cam_thread = threading.Thread(target=loop, daemon=True)
        self._cam_thread.start()
        self.btn_cam.configure(text="Stop camera")
        self.lbl_cam_status.configure(text="Streaming… buffer fills automatically (max 64)")

    def _clear_buffer(self) -> None:
        with self._buffer_lock:
            self.demo.frame_buffer.clear()
            self.demo.last_prediction = ""
            self.demo.last_translation = ""
        self.txt_gloss.delete("1.0", self.tk.END)
        self.txt_en.delete("1.0", self.tk.END)

    def _decode_sign(self) -> None:
        def work() -> None:
            with self._buffer_lock:
                n = len(self.demo.frame_buffer)
                if n < 16:
                    self.root.after(
                        0,
                        lambda: self.messagebox.showinfo(
                            "Buffer", f"Need at least 16 frames in buffer (have {n})."
                        ),
                    )
                    return
                result = self.demo.predict()
            if not result:
                self.root.after(
                    0,
                    lambda: self.messagebox.showinfo("Decode", "No prediction returned."),
                )
                return
            gloss_line, en = result

            def apply() -> None:
                self.txt_gloss.delete("1.0", self.tk.END)
                self.txt_gloss.insert(self.tk.END, gloss_line or "")
                self.txt_en.delete("1.0", self.tk.END)
                self.txt_en.insert(self.tk.END, en or "")

            self.root.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def _load_pipeline_async(self) -> None:
        self.lbl_pipe.configure(text="Loading…")

        def work() -> None:
            pipe, err = load_speech_to_sign(self.project_root)
            self._pipeline = pipe
            self._pipeline_err = err

            def done() -> None:
                if err:
                    self.lbl_pipe.configure(text="Failed")
                    self.messagebox.showerror("Pipeline", err)
                else:
                    self.lbl_pipe.configure(text="Ready")

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _stop_sign_playback(self) -> None:
        if self._after_play_id is not None:
            try:
                self.root.after_cancel(self._after_play_id)
            except Exception:
                pass
            self._after_play_id = None

    def _generate_sign_video(self) -> None:
        if self._pipeline is None:
            if self._pipeline_err:
                self.messagebox.showerror("Pipeline", self._pipeline_err)
            else:
                self.messagebox.showinfo(
                    "Pipeline", 'Click "Load text→sign pipeline" first.'
                )
            return

        text = self.entry_de.get().strip()
        if not text:
            return

        self._stop_sign_playback()

        def work() -> None:
            try:
                out = self._pipeline(text, return_intermediates=True)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self.messagebox.showerror("Generate", str(e)),
                )
                return

            video = out.get("video")
            glosses = out.get("glosses") or []

            def done() -> None:
                self.txt_gloss_out.delete("1.0", self.tk.END)
                self.txt_gloss_out.insert(self.tk.END, " ".join(glosses))
                if video is None or len(video) == 0:
                    self.messagebox.showinfo("Generate", "No video frames returned.")
                    return
                self._sign_video_frames = video
                self._sign_video_idx = 0
                self._play_sign_frame()

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _play_sign_frame(self) -> None:
        if self._sign_video_frames is None:
            return
        frames = self._sign_video_frames
        i = self._sign_video_idx % len(frames)
        fr = frames[i]
        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)
        h, w = fr.shape[:2]
        max_w = 420
        if w > max_w:
            scale = max_w / float(w)
            nh = int(h * scale)
            fr = cv2.resize(fr, (max_w, nh), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(fr)
        photo = ImageTk.PhotoImage(image=im)
        self._latest_sign_photo = photo
        self.lbl_sign.configure(image=photo)
        self._sign_video_idx = i + 1
        self._after_play_id = self.root.after(40, self._play_sign_frame)

    def _on_close(self) -> None:
        self._stop_sign_playback()
        self._cam_stop.set()
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=2.0)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidirectional SLR desktop demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/e2e/best.pth",
        help="Hybrid SLR checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root (PHOENIX data + checkpoints)",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    ckpt = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    proj = (ROOT / args.root).resolve() if args.root == "." else Path(args.root).resolve()

    app = BidirectionalDemoApp(ckpt, args.device, proj, args.camera)
    app.run()


if __name__ == "__main__":
    main()
