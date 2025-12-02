#!/usr/bin/env python3

import os
import re

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import Sam3VideoModel, Sam3VideoProcessor


def get_session(model, processor, frames, prompts, device):
    session = processor.init_video_session(video=frames, inference_device=device, dtype=torch.bfloat16)
    for text in prompts:
        session = processor.add_text_prompt(session, text=text)
    return session


def main():
    # --- 1. Setup Paths & Config ---
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])    
    min_score = float(os.environ["SAM3_MIN_SCORE"])
    min_percent = float(os.environ["SAM3_MIN_FRAME_PERCENT"])
    device = "cuda"

    img_dir = data_root / scene / "images"
    out_dir = data_root / scene / "instance_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load & Sort Frames ---
    prompts = [p.strip() for p in Path("prompts.txt").read_text().splitlines() if p.strip()]
    frame_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg'}],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    
    video_frames = [Image.open(f) for f in frame_files]
    h, w = video_frames[0].size[::-1]
    
    print(f"[SAM3] Scene: {scene} | Frames: {len(frame_files)}")
    print(f"Loading Model on {device}...")
    
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # --- 3. Pass 1: Count Objects ---
    print("Pass 1/2: Analyzing Object Stability...")
    obj_counts = Counter()
    
    session = get_session(model, processor, video_frames, prompts, device)
    
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(session):
            processed = processor.postprocess_outputs(session, output)
            
            valid_mask = processed["scores"].cpu().numpy() >= min_score
            valid_ids = processed["object_ids"][valid_mask].cpu().numpy().astype(int)
            
            obj_counts.update(valid_ids)

    min_frames = int(np.ceil((min_percent / 100.0) * len(frame_files)))
    valid_obj_ids = {oid for oid, count in obj_counts.items() if count >= min_frames}
    
    print(f"[SAM3] Found {len(obj_counts)} objects. Keeping {len(valid_obj_ids)} valid ones.")
    if not valid_obj_ids: return

    id_map = {oid: i for i, oid in enumerate(sorted(valid_obj_ids)) if i < 255}

    # --- 4. Pass 2: Save Masks ---
    print("Pass 2/2: Generating Masks...")
    
    session = get_session(model, processor, video_frames, prompts, device)

    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(session):
            processed = processor.postprocess_outputs(session, output)
            
            masks = processed["masks"].cpu().numpy().squeeze()
            scores = processed["scores"].cpu().numpy()
            obj_ids = processed["object_ids"].cpu().numpy().astype(int)

            if masks.ndim == 2: masks = masks[np.newaxis, ...]

            final_mask = np.full((h, w), 255, dtype=np.uint8)

            valid_indices = [
                i for i, s in enumerate(scores) 
                if s >= min_score and obj_ids[i] in id_map
            ]

            if valid_indices:
                current_masks = masks[valid_indices]
                areas = current_masks.sum(axis=(1, 2))
                
                sorted_local_indices = np.argsort(areas)[::-1]

                for idx in sorted_local_indices:
                    real_idx = valid_indices[idx]
                    oid = obj_ids[real_idx]
                    
                    label = id_map[oid]
                    final_mask[current_masks[idx] > 0] = label

            save_name = frame_files[output.frame_idx].with_suffix(".png").name
            Image.fromarray(final_mask).save(out_dir / save_name)

    print(f"[SAM3] Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()