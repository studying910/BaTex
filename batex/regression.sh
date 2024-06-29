# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/root/autodl-tmp/stable_diffusion/stable-diffusion-v1-5"
export DATA_DIR="/root/autodl-tmp/textual_inversion/data/cat"

accelerate launch --config_file config-1.yaml --main_process_port 26000 textual_inversion_regression.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<custom_cat>" --initializer_token="cat" \
  --dist_type="Vector_dot" --mask_k=720 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/root/autodl-tmp/textual_inversion/trained_embeddings/custom_cat/regression_720_test" \
  --only_save_embeds \
  --checkpointing_steps 50000 \
  --save_steps 50000 \
  --test_score --score_steps 50 \
  # --validation_prompt "a photo of <custom_cat>" \
  # --num_validation_images 4 \
  # --validation_steps 50 \
  # --seed 512 \
  # --resume_from_checkpoint "latest"