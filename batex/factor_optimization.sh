# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/root/autodl-tmp/stable_diffusion/stable-diffusion-v1-5"
export DATA_DIR="/root/autodl-tmp/textual_inversion/data/barn"

accelerate launch --config_file config.yaml factor_optimization.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --pivotal_recon_dir "/root/autodl-tmp/textual_inversion/trained_embeddings/custom_barn/pivotal_reconstruction"
  --placeholder_token_content="<custom_barn_content>" \
  --placeholder_token_style="<custom_barn_style>" \
  --lambda_content 0.1 --lambda_style 0.1 \
  --blocks_content 1 2 3 4 --blocks_style 1 2 3 4 \
  --weights_content 1.0 1.0 1.0 1.0 --weights_style 1.0 1.0 1.0 1.0 \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/root/autodl-tmp/textual_inversion/trained_embeddings/custom_cat/factor_optimization" \
  --only_save_embeds \
  --validation_prompt "a photo of content <custom_barn_content> in the style of <custom_cat_barn>" \
  --num_validation_images 4 \
  --validation_steps 100 \
  # --resume_from_checkpoint "latest"