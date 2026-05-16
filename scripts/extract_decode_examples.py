import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.phoenix_dataset import PhoenixDataset, collate_fn  # noqa: E402
from models.transformer import SignLanguageTransformer, HybridCTCAttentionModel  # noqa: E402
from evaluate import ctc_decode  # noqa: E402


def edit_distance(pred: List[str], target: List[str]) -> int:
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def load_json(path: Path):
    return json.loads(path.read_text())


def decode_transformer_samples(checkpoint_dir: Path, split: str, max_samples: int, device: torch.device):
    config = load_json(checkpoint_dir / "config.json")
    vocab = load_json(checkpoint_dir / "vocab.json")
    idx2gloss = {v: k for k, v in vocab.items()}

    dataset = PhoenixDataset(
        str(ROOT / "data" / "phoenix2014-release"),
        split=split,
        max_frames=64,
        load_video=True,
        vocab=vocab,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = SignLanguageTransformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        use_cnn_backbone=config.get("use_cnn_backbone", True),
        cnn_type=config.get("cnn_type", "resnet"),
    ).to(device)
    ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    blank_idx = vocab.get("<blank>", 2)
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            frame_lengths = batch["frame_lengths"]
            log_probs = model(frames)
            preds = ctc_decode(log_probs, frame_lengths, idx2gloss, blank_idx)
            for p, t in zip(preds, batch["glosses"]):
                rows.append({"target": t, "pred": p})
                if len(rows) >= max_samples:
                    return rows
    return rows


def decode_hybrid_samples(checkpoint_dir: Path, split: str, max_samples: int, device: torch.device):
    config = load_json(checkpoint_dir / "config.json")
    vocab = load_json(checkpoint_dir / "vocab.json")
    idx2gloss = {v: k for k, v in vocab.items()}

    dataset = PhoenixDataset(
        str(ROOT / "data" / "phoenix2014-release"),
        split=split,
        max_frames=64,
        load_video=True,
        vocab=vocab,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = HybridCTCAttentionModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        ctc_weight=config.get("ctc_weight", 0.3),
        use_resnet=True,
    ).to(device)
    ckpt = torch.load(checkpoint_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    blank_idx = vocab.get("<blank>", 2)
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            frame_lengths = batch["frame_lengths"]
            ctc_log_probs, _, _ = model(frames)
            preds = ctc_decode(ctc_log_probs, frame_lengths, idx2gloss, blank_idx)
            for p, t in zip(preds, batch["glosses"]):
                rows.append({"target": t, "pred": p})
                if len(rows) >= max_samples:
                    return rows
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = "test"
    max_samples = 40

    transformer_rows = decode_transformer_samples(ROOT / "checkpoints" / "transformer", split, max_samples, device)
    hybrid_rows = decode_hybrid_samples(ROOT / "checkpoints" / "hybrid", split, max_samples, device)

    merged = []
    for i, (a, b) in enumerate(zip(transformer_rows, hybrid_rows)):
        target = a["target"]
        # Guard against occasional mismatch due to vocabulary-specific preprocessing differences.
        if target != b["target"]:
            target = b["target"]
        ctc_pred = a["pred"]
        hybrid_pred = b["pred"]
        merged.append(
            {
                "index": i,
                "target": " ".join(target),
                "ctc_baseline_pred": " ".join(ctc_pred),
                "hybrid_pred": " ".join(hybrid_pred),
                "ctc_edit_distance": edit_distance(ctc_pred, target),
                "hybrid_edit_distance": edit_distance(hybrid_pred, target),
            }
        )

    improved = [r for r in merged if r["hybrid_edit_distance"] < r["ctc_edit_distance"]]
    selected = improved[0] if improved else (merged[0] if merged else None)

    out_dir = ROOT / "documentation" / "thesis_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decode_examples_ctc_vs_hybrid.json"
    out_path.write_text(json.dumps({"split": split, "samples": merged, "selected_example": selected}, indent=2))
    print(f"Wrote decode evidence: {out_path}")
    if selected:
        print("Selected example:")
        print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
