#!/bin/bash

set -e


# --- Configuration ---
SCENE_NAME="${SCENE_NAME}"
CACHE_ROOT="/app/holoscene"
DATA_ROOT="${DATA_ROOT}"
DATA_DIR="${DATA_ROOT}/${SCENE_NAME}"

WONDER3D_DIR="${CACHE_ROOT}/ckpts"
LAMA_DIR="${CACHE_ROOT}/lama"
LAMA_CHECK_FILE="$LAMA_DIR/big-lama/models/best.ckpt"
OMNIDATA_FILE="${CACHE_ROOT}/omnidata/omnidata_dpt_normal_v2.ckpt"


# --- Check Wonder3D+ models ---
echo "--- Checking for Wonder3D+ models... ---"
if [ ! -d "$WONDER3D_DIR" ] || [ -z "$(ls -A $WONDER3D_DIR)" ]; then
    echo "--- Wonder3D+ models not found. Downloading... ---"
    mkdir -p "$WONDER3D_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir='$WONDER3D_DIR')
"
    echo "--- Wonder3D+ download complete. ---"
else
    echo "--- Found existing Wonder3D+ models. ---"
fi


# --- Check LaMa model ---
echo "--- Checking for LaMa model... ---"
if [ ! -f "$LAMA_CHECK_FILE" ]; then
    echo "--- LaMa model not found. Downloading... ---"
    mkdir -p "$LAMA_DIR"
    
    pushd "$LAMA_DIR" > /dev/null
        curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
        unzip -o big-lama.zip
        rm big-lama.zip
    popd > /dev/null
    
    echo "--- LaMa download complete. ---"
else
    echo "--- Found existing LaMa model. ---"
fi


# --- Check Omnidata model ---
echo "--- Checking for Omnidata model... ---"
if [ ! -f "$OMNIDATA_FILE" ]; then
    echo "--- Omnidata model not found. Downloading... ---"
    mkdir -p "$(dirname "$OMNIDATA_FILE")"
    
    wget -O "$OMNIDATA_FILE" "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt"
    echo "--- Omnidata download complete. ---"
else
    echo "--- Found existing Omnidata model. ---"
fi

echo "--- All models are cached. ---"


############################################
# Create custom configs for all stages
############################################
echo "--- Creating custom config files... ---"
CONFIG_DIR="/app/holoscene/confs/custom"
mkdir -p "$CONFIG_DIR"

BASE_CONF="${CONFIG_DIR}/${SCENE_NAME}.conf"
POST_CONF="${CONFIG_DIR}/${SCENE_NAME}_post.conf"
TEX_CONF="${CONFIG_DIR}/${SCENE_NAME}_tex.conf"

cp /app/holoscene/confs/replica/room_0/replica_room_0.conf "$BASE_CONF"
cp /app/holoscene/confs/replica/room_0/replica_room_0_post.conf "$POST_CONF"
cp /app/holoscene/confs/replica/room_0/replica_room_0_tex.conf "$TEX_CONF"

for CFG in "$BASE_CONF" "$POST_CONF" "$TEX_CONF"; do
    sed -i "s|data_root_dir = ./data_dir/replica/|data_root_dir = ${DATA_ROOT}/|g" "$CFG"
    sed -i "s|data_dir = room_0|data_dir = ${SCENE_NAME}|g" "$CFG"
    sed -i "s|expname = holoscene_replica_room_0|expname = holoscene_${SCENE_NAME}|g" "$CFG"
done

echo "--- Custom configs created:"
echo "    Stage 1: $BASE_CONF"
echo "    Stage 2: $POST_CONF"
echo "    Stage 3/4: $TEX_CONF"
echo "----------------------------------------"


############################################
# Stage 0: Priors
############################################
echo "--- Running Stage 0: Generating priors... ---"
python3 marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir="${DATA_DIR}/images" \
    --output_dir="${DATA_DIR}/"

python3 marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir="${DATA_DIR}/images" \
    --output_dir="${DATA_DIR}/"
echo "--- Stage 0 complete. ---"


############################################
# Stage 1: Initial reconstruction
############################################
echo "--- Running Stage 1: Initial reconstruction... ---"
python3 training/exp_runner.py --conf "$BASE_CONF"


############################################
# Stage 2: Post-processing
############################################
echo "--- Running Stage 2: Post-processing... ---"
python3 training/exp_runner_post.py --conf "$POST_CONF" \
    --is_continue \
    --timestamp latest \
    --checkpoint latest


############################################
# Stage 3: Texture refinement
############################################
echo "--- Running Stage 3: Texture refinement... ---"
python3 training/exp_runner_texture.py --conf "$TEX_CONF" \
    --is_continue \
    --timestamp latest \
    --checkpoint latest


############################################
# Stage 4: Gaussian on mesh
############################################
echo "--- Running Stage 4: Gaussian on mesh... ---"
python3 training/exp_runner_gaussian_on_mesh.py --conf "$TEX_CONF" \
    --is_continue \
    --timestamp latest \
    --checkpoint latest


############################################
# Export
############################################
echo "--- Training stages complete, exporting results... ---"

python3 export/export_glb.py --conf "$TEX_CONF" --timestamp latest
python3 export/export_usd.py --conf "$TEX_CONF" --timestamp latest
python3 export/export_gs_usd.py --conf "$TEX_CONF" --timestamp latest

echo "--- Export complete. ---"