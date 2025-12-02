#!/usr/bin/env python3

import os
import re
import json

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from depth_anything_3.api import DepthAnything3


def main():
    # --- 1. Setup Paths & Config ---
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    model_dir = os.environ["DA3_MODEL_DIR"]
    device = "cuda"
    
    img_dir = data_root / scene / "images"
    out_json = data_root / scene / "transforms.json"
    cache_dir = data_root / scene
    
    # --- 2. Load & Sort Frames ---
    frame_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )

    print(f"[DA3] Scene='{scene}', Frames={len(frame_files)}")

    with Image.open(frame_files[0]) as img:
        orig_w, orig_h = img.size
    print(f"[DA3] Original Res: {orig_w}x{orig_h}")

    # --- 3. Load Model & Run Inference ---
    print(f"[DA3] Loading model from {model_dir}...")
    
    model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

    frame_paths_str = [str(f) for f in frame_files]
    
    with torch.no_grad():
        preds = model.inference(
            image=frame_paths_str,
            export_dir=str(cache_dir),
            export_format="npz"
        )

    # --- 3. Vectorized Intrinsics Processing ---
    proc_h, proc_w = preds.processed_images.shape[1:3]
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h

    Ks = preds.intrinsics.astype(np.float32)
    
    fx = Ks[:, 0, 0] * scale_x
    fy = Ks[:, 1, 1] * scale_y
    cx = Ks[:, 0, 2] * scale_x
    cy = Ks[:, 1, 2] * scale_y

    params = np.stack([fx, fy, cx, cy], axis=1)
    valid_mask = np.isfinite(params).all(axis=1) & (params[:, 0] > 1e-3)
    valid_params = params[valid_mask]

    final_fx, final_fy, final_cx, final_cy = np.median(valid_params, axis=0)
    angle_x = 2.0 * np.arctan((orig_w / 2.0) / final_fx)

    # --- 4. Vectorized Extrinsics Processing ---
    num_frames = len(preds.extrinsics)
    
    w2c = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(num_frames, axis=0)
    w2c[:, :3, :] = preds.extrinsics

    c2w_all = np.linalg.inv(w2c)

    # --- 5. Generate JSON ---
    frames_json = [
        {
            "file_path": f"images/{f.name}",
            "transform_matrix": c2w.tolist()
        }
        for f, c2w in zip(frame_files, c2w_all)
    ]

    out_data = {
        "camera_angle_x": float(angle_x),
        "fl_x": float(final_fx),
        "fl_y": float(final_fy),
        "cx": float(final_cx),
        "cy": float(final_cy),
        "w": int(orig_w),
        "h": int(orig_h),
        "frames": frames_json,
    }

    out_json.write_text(json.dumps(out_data, indent=4))
    print(f"[DA3] Done. Saved to {out_json}")


if __name__ == "__main__":
    main()