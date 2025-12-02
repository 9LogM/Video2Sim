FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    cmake \
    unzip \
    python3-tk \
    python3-pip \
    python3-dev \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
RUN pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@24abe714105441f049b50be1fc8eb411d727e6e6
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8#egg=nvdiffrast
RUN sed -i '/TORCH_CUDA_ARCH_LIST/s/^/#/' /usr/local/lib/python3.10/dist-packages/nvdiffrast/torch/ops.py
RUN MAX_JOBS=15 pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
RUN TCNN_CUDA_ARCHITECTURES=$(echo $TORCH_CUDA_ARCH_LIST | tr -d .) pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
RUN pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com

WORKDIR /app/holoscene
RUN git clone https://github.com/xiahongchi/HoloScene.git .
RUN pip install -r requirements.txt
RUN sed -i 's/self.glctx = dr.RasterizeGLContext()/self.glctx = dr.RasterizeCudaContext()/g' training/holoscene_train.py

COPY hs_process.sh .