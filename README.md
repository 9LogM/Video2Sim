# Video2Sim

Video2Sim is a Docker-based pipeline for converting raw video (or sensor logs)
into simulation-ready 3D assets using a sequence of modular components.


## Requirements
- NVIDIA GPU with CUDA support
- 16GB VRAM
- 200GB Disk Space
- Docker Engine
- A Hugging Face access token with permission to pull from:
  - `https://huggingface.co/facebook/sam3`


## Pipeline Overview

### Preprocessor

Takes a video or ROS bag and outputs images.

### DA3 (Depth Anything 3)

Reads extracted frames and produces a NeRF‑style `transforms.json`
containing camera intrinsics and per-frame camera poses.

### SAM 3 (Segment Anything Model 3)

Reads extracted frames and produces consistent per‑frame instance masks.
Prompts are defined in `prompts.txt`.

### HoloScene

Uses extracted frames, poses, masks, and Marigold‑generated priors
to reconstruct the 3D scene and export final assets.


## Running the Pipeline

1. **Place input files**  
Place `.mp4` or `.bag` into `data/input`.

2. **Configure environment**  
Fill out `.env` (scene name, Hugging Face token, etc.).

3. **Build and run each module in sequence**  
    ```bash
    docker compose up --build preprocessor
    docker compose up --build da3
    docker compose up --build sam3
    docker compose up --build holoscene
    ```

4. **Retrieve results**  
Results will be generated into `data/output`.

**Note:** To keep the container alive for debugging, temporarily set
the command in `docker-compose.yml` to `["tail", "-f", "/dev/null"]`."


## Citations

```bibtex
@article{depthanything3,
        title       = {Depth Anything 3: Recovering the visual space from any views},
        author      = {Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
        journal     = {arXiv preprint arXiv:2511.10647},
        year        = {2025}
}
```

```bibtex
@misc{carion2025sam3segmentconcepts,
      title         = {SAM 3: Segment Anything with Concepts},
      author        = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year          = {2025},
      eprint        = {2511.16719},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2511.16719},
}
```

```bibtex
@misc{xia2025holoscene,
      title         = {HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video}, 
      author        = {Hongchi Xia and Chih-Hao Lin and Hao-Yu Hsu and Quentin Leboutet and Katelyn Gao and Michael Paulitsch and Benjamin Ummenhofer and Shenlong Wang},
      year          = {2025},
      eprint        = {2510.05560},
      archivePrefi  = {arXiv},
      primaryClas   = {cs.CV},
      url           = {https://arxiv.org/abs/2510.05560}, 
}
```