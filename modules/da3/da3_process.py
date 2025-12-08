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
        "model_dir": "depth-anything/DA3NESTED-GIANT-LARGE",
        "device": "cuda"
    }


def compute_intrinsics(preds, orig_w, orig_h):
    """Scales intrinsics to original resolution, computes median values, and derives FOV."""
    _, proc_h, proc_w = preds.depth.shape 
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


def process_poses(preds):
    """
    1. Invert (W2C -> C2W): Convert World-to-Camera matrices to Camera-to-World poses.
    2. CV -> GL: Convert camera coordinate system (OpenCV: Down/Right) to OpenGL (Up/Back).
    3. Center: Subtract the mean translation to center the trajectory at (0,0,0).
    4. Align: Use SVD/PCA to align the trajectory's principal axes to the world axes.
    5. Reorient world: apply 180° rotation around the X axis.
    """
    ext = preds.extrinsics.astype(np.float32)
    n = ext.shape[0]

    R = ext[:, :3, :3]
    t = ext[:, :3, 3] 
    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -(R_inv @ t[..., None])[..., 0]
    c2w = np.eye(4, dtype=np.float32)[None].repeat(n, axis=0)
    c2w[:, :3, :3] = R_inv
    c2w[:, :3, 3] = t_inv

    flip_cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    c2w = c2w @ flip_cv_to_gl

    c2w[:, :3, 3] -= c2w[:, :3, 3].mean(axis=0)

    _, _, vh = np.linalg.svd(c2w[:, :3, 3])
    R_align = vh
    avg_cam_y = c2w[:, :3, 1].mean(axis=0) 
    if np.dot(R_align[2, :], avg_cam_y) > 0: 
        R_align[2, :] *= -1
        R_align[1, :] *= -1
    c2w[:, :3, :3] = R_align @ c2w[:, :3, :3]
    c2w[:, :3, 3] = (R_align @ c2w[:, :3, 3].T).T

    floor_fix_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    c2w = floor_fix_matrix @ c2w

    return c2w


def main():
    cfg = get_config()
    
    files = sorted(
        [f for f in cfg["img_dir"].iterdir() if not f.name.startswith('.')],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    
    with Image.open(files[0]) as img:
        orig_w, orig_h = img.size

    print(f"[DA3] Scene: {cfg['scene']} | Frames: {len(files)} | Res: {orig_w}x{orig_h}")
    
    model = DepthAnything3.from_pretrained(cfg["model_dir"]).to(cfg["device"])
    
    print(f"Running inference...")
    with torch.inference_mode(), torch.autocast(device_type=cfg["device"], dtype=torch.bfloat16):
        preds = model.inference(
            image=[str(f) for f in files],
            export_dir=str(cfg["out_dir"]),
            export_format="glb",
            use_ray_pose=True
        )

    fx, fy, cx, cy, angle_x = compute_intrinsics(preds, orig_w, orig_h)
    c2w_matrices = process_poses(preds)

    frames_json = [
        {"file_path": f"images/{f.name}", "transform_matrix": m.tolist()}
        for f, m in zip(files, c2w_matrices)
    ]

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