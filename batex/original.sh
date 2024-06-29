# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/root/autodl-tmp/stable_diffusion/stable-diffusion-v1-5"
export DATA_DIR="/root/autodl-tmp/textual_inversion/datasets/colorful_teapot"

accelerate launch --config_file config.yaml textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<colorful_teapot>" --initializer_token="pot" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/root/autodl-tmp/textual_inversion/trained_embeddings/colorful_teapot/original" \
  --only_save_embeds \
  --validation_prompt "a photo of <colorful_teapot>" \
  --num_validation_images 4 \
  --validation_steps 500 \
  # --resume_from_checkpoint "latest"