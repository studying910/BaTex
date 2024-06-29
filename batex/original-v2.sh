export MODEL_NAME="/root/autodl-tmp/stable_diffusion/stable-diffusion-2"
export DATA_DIR="/root/autodl-tmp/textual_inversion/data/cat"

accelerate launch --config_file config.yaml textual_inversion_v2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<custom_cat>" --initializer_token="cat" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/root/autodl-tmp/textual_inversion/trained_embeddings/custom_cat_v2/original" \
  --only_save_embeds \
  --validation_prompt "a photo of <custom_cat>" \
  --num_validation_images 4 \
  --validation_steps 500 \
  # --resume_from_checkpoint "latest"