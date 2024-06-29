# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/root/autodl-tmp/stable_diffusion/stable-diffusion-v1-5"
export DATA_DIR="/root/autodl-tmp/textual_inversion/data/chair"

accelerate launch --config_file config.yaml pivotal_reconstruction.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token_content="<custom_chair_content>" --initializer_token_content="chair" \
  --placeholder_token_style="<custom_chair_style>" --initializer_token_style="blue" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/root/autodl-tmp/textual_inversion/trained_embeddings/custom_chair/pivotal_reconstruction" \
  --only_save_embeds \
  --validation_prompt "a photo of content <custom_chair_content> in the style of <custom_chair_style>" \
  --num_validation_images 4 \
  --validation_steps 500 \
  # --resume_from_checkpoint "latest"