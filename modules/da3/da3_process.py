#!/usr/bin/env python3

import os
import re
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from depth_anything_3.api import DepthAnything3


def get_config():
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    output_root = Path(os.environ["OUTPUT_ROOT"])
    return {
        "scene": scene,
        "img_dir": data_root / scene / "images",
        "out_json": data_root / scene / "transforms.json",
        "out_dir": output_root / scene / "da3_out",
        "model_dir": os.environ["DA3_MODEL_DIR"],
        "device": "cuda"
    }


def get_frame_paths(img_dir):
    """Returns sorted frame paths."""
    files = sorted(
        [f for f in img_dir.iterdir() if not f.name.startswith('.')],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    return files


def get_image_dims(file_path):
    """Peeks at the first image to get dimensions."""
    with Image.open(file_path) as img:
        return img.size


def get_processed_hw(preds):
    """Extracts processed height and width from model predictions."""
    s = preds.processed_images.shape
    if len(s) == 4 and s[1] == 3:
        return s[2], s[3]
    if len(s) == 4 and s[-1] == 3:
        return s[1], s[2]


def compute_intrinsics(preds, orig_w, orig_h):
    """Scales intrinsics back to original resolution."""
    proc_h, proc_w = get_processed_hw(preds)
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h

    Ks = preds.intrinsics.astype(np.float32)
    fx = Ks[:, 0, 0] * scale_x
    fy = Ks[:, 1, 1] * scale_y
    cx = Ks[:, 0, 2] * scale_x
    cy = Ks[:, 1, 2] * scale_y

    params = np.stack([fx, fy, cx, cy], axis=1)
    valid_mask = np.isfinite(params).all(axis=1) & (params[:, 0] > 1e-3)
    
    if not valid_mask.any():
        raise RuntimeError("No valid intrinsics found.")

    final_fx, final_fy, final_cx, final_cy = np.median(params[valid_mask], axis=0)
    angle_x = 2.0 * np.arctan((orig_w / 2.0) / final_fx)

    return final_fx, final_fy, final_cx, final_cy, angle_x


def compute_extrinsics(preds):
    """Ensures extrinsics are valid 4x4 matrices."""
    ext = preds.extrinsics.astype(np.float32)
    n = ext.shape[0]

    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    if ext.shape[-2:] == (3, 4):
        w2c[:, :3, :] = ext
    elif ext.shape[-2:] == (4, 4):
        w2c[:] = ext
    else:
        raise ValueError(f"Unexpected extrinsics shape: {ext.shape}")

    c2w = np.linalg.inv(w2c)
    return c2w


def normalize_cameras(c2w):
    """
    Centers the scene and aligns the dataset principal axes to World X, Y, Z.
    """
    c2w[:, :3, 3] -= c2w[:, :3, 3].mean(axis=0)

    _, _, vh = np.linalg.svd(c2w[:, :3, 3])
    R_new = vh

    avg_cam_y = c2w[:, :3, 1].mean(axis=0) 
    if np.dot(R_new[2, :], avg_cam_y) > 0: 
        R_new[2, :] *= -1
        R_new[1, :] *= -1

    c2w[:, :3, :3] = R_new @ c2w[:, :3, :3]
    c2w[:, :3, 3] = (R_new @ c2w[:, :3, 3].T).T
    
    return c2w


def main():
    cfg = get_config()
    frame_files = get_frame_paths(cfg["img_dir"])
    orig_w, orig_h = get_image_dims(frame_files[0])

    print(f"[DA3] Scene: {cfg['scene']} | Frames: {len(frame_files)} | Res: {orig_w}x{orig_h}")

    print(f"[DA3] Loading model from {cfg['model_dir']}...")
    model = DepthAnything3.from_pretrained(cfg["model_dir"]).to(cfg["device"])
    
    print("Running inference...")
    with torch.inference_mode(), torch.autocast(device_type=cfg["device"], dtype=torch.bfloat16):
        preds = model.inference(
            image=[str(f) for f in frame_files],
            export_dir=str(cfg["out_dir"]),
            export_format="glb",
            use_ray_pose=True
        )

    fx, fy, cx, cy, angle_x = compute_intrinsics(preds, orig_w, orig_h)
    c2w_matrices = compute_extrinsics(preds)

    c2w_matrices = normalize_cameras(c2w_matrices)

    print(f"\n[Stats] Camera Parameters:")
    print(f"  Focal Length (X/Y): {fx:.2f} / {fy:.2f}")
    print(f"  Principal Point   : {cx:.2f}, {cy:.2f}")
    print(f"  Field of View (X) : {np.degrees(angle_x):.2f} deg\n")

    frames_json = []
    for f, matrix in zip(frame_files, c2w_matrices):
        frames_json.append({
            "file_path": f"images/{f.name}",
            "transform_matrix": matrix.tolist()
        })

    out_data = {
        "camera_angle_x": float(angle_x),
        "fl_x": float(fx), "fl_y": float(fy),
        "cx": float(cx),   "cy": float(cy),
        "w": int(orig_w),  "h": int(orig_h),
        "frames": frames_json,
    }

    cfg["out_json"].write_text(json.dumps(out_data, indent=4))
    print(f"[DA3] Done. Saved transforms to {cfg['out_json']}")


if __name__ == "__main__":
    main()