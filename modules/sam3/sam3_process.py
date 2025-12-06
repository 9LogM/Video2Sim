#!/usr/bin/env python3

import os
import re
import torch
import numpy as np

from PIL import Image
from pathlib import Path
from collections import Counter
from transformers import (
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
)


def main():
    # --- 1. Config & Setup ---
    scene = os.environ["SCENE_NAME"]
    data_root = Path(os.environ["DATA_ROOT"])
    min_score = float(os.environ["SAM3_MIN_SCORE"])
    min_duration = float(os.environ["SAM3_MIN_FRAME_PERCENT"]) / 100.0
    device = "cuda"

    img_dir = data_root / scene / "images"
    out_dir = data_root / scene / "instance_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [p.strip() for p in Path("prompts.txt").read_text().splitlines() if p.strip()]
    
    frame_files = sorted(
        [f for f in img_dir.iterdir() if f.name[0] != '.'],
        key=lambda x: int(re.search(r"\d+", x.name).group() or 0)
    )
    
    frames = [Image.open(f) for f in frame_files]
    h, w = frames[0].size[::-1]

    print(f"[SAM3] Processing {len(frames)} frames. Score > {min_score}, Frame Duration > {min_duration}%")

    # --- 2. Load Model & Run Inference ---  
    dtype = torch.bfloat16
    
    pcs_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    pcs_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    
    pvs_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    pvs_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

    # --- 3. Bootstrap: Get Seed Masks ---
    pcs_session = pcs_processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    pcs_session = pcs_processor.add_text_prompt(pcs_session, prompts)

    init_masks, init_ids = [], []
    
    with torch.inference_mode():
        for output in pcs_model.propagate_in_video_iterator(pcs_session):
            if output.frame_idx > 0: break
            
            processed = pcs_processor.postprocess_outputs(pcs_session, output)
            scores = processed["scores"].cpu().numpy()
            
            keep = scores >= min_score
            init_ids = processed["object_ids"].cpu().numpy()[keep].astype(int).tolist()
            init_masks = [m for m in processed["masks"].cpu().numpy()[keep].astype(np.uint8)]
            break

    if not init_ids:
        print("No objects found to track.")
        return

    # --- 4. Setup Tracker ---
    print(f"Seeding tracker with {len(init_ids)} objects...")
    pvs_session = pvs_processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )

    pvs_processor.add_inputs_to_inference_session(
        pvs_session, frame_idx=0, input_masks=init_masks, obj_ids=init_ids
    )

    with torch.inference_mode():
        _ = pvs_model(inference_session=pvs_session, frame_idx=0)

    # --- 5. Run Propagation ---
    print("Running PVS inference...")
    frame_results = []
    id_counter = Counter()

    with torch.inference_mode():
        for output in pvs_model.propagate_in_video_iterator(pvs_session):
            masks_tensor = pvs_processor.post_process_masks(
                [output.pred_masks], original_sizes=[(h, w)], binarize=True
            )[0]
            
            if isinstance(masks_tensor, torch.Tensor):
                masks_tensor = masks_tensor.detach().to(torch.float32).cpu().numpy()
            
            masks = np.array(masks_tensor).astype(np.uint8)
            ids = np.array(list(pvs_session.obj_ids), dtype=int)

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0, :, :]
            elif masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            
            n = min(len(ids), len(masks))
            ids, masks = ids[:n], masks[:n]

            for oid, m in zip(ids, masks):
                if m.sum() > 0: id_counter.update([oid])

            frame_results.append({
                "file_name": frame_files[output.frame_idx].name,
                "ids": ids,
                "masks": masks
            })

    # --- 6. Filter & Save ---
    min_frames = np.ceil(len(frames) * min_duration)
    valid_ids = {oid for oid, c in id_counter.items() if c >= min_frames}
    
    id_map = {oid: i for i, oid in enumerate(sorted(valid_ids)) if i < 255}

    print(f"Saving masks for {len(id_map)} objects...")

    for res in frame_results:
        canvas = np.full((h, w), 255, dtype=np.uint8)
        
        objects_to_draw = []
        for i, oid in enumerate(res["ids"]):
            if oid in id_map:
                mask = res["masks"][i]
                objects_to_draw.append((mask, oid, mask.sum()))
        
        objects_to_draw.sort(key=lambda x: x[2], reverse=True)

        for mask, oid, _ in objects_to_draw:
            canvas[mask > 0] = id_map[oid]

        Image.fromarray(canvas).save(out_dir / res["file_name"])

    print("Done.")


if __name__ == "__main__":
    main()