#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import warnings
import tqdm as tq
from pathlib import Path
from typing import Optional
from shutil import rmtree

import numpy as np
import PIL
import json
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import sklearn.preprocessing
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0.dev0")

logger = get_logger(__name__)


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    '''
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    '''

    validation_path = os.path.join(args.output_dir, "validation_images")
    os.makedirs(validation_path, exist_ok=True)
    for i in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=50, guidance_scale=7.5,
                             generator=generator).images[0]
            validation_image_path = f"{validation_path}/step-{global_step}_{i + 1}.jpeg"
            image.save(validation_image_path)

    del pipeline
    torch.cuda.empty_cache()


def save_progress(text_encoder, placeholder_token_id, weight_vector, candidate_embedding_matrix, vocab_ids, accelerator,
                  args, save_path):
    logger.info("Saving learned embedding, weight vector and candidate matrix")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    weight_vector = accelerator.unwrap_model(weight_vector).weight
    candidate_embedding_matrix = accelerator.unwrap_model(candidate_embedding_matrix)
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu(),
                           "weight_vector": weight_vector.detach().cpu(),
                           "candidate_embedding_matrix": candidate_embedding_matrix.detach().cpu(),
                           "vocab_ids": vocab_ids.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def score_computation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step,
                      score_i2i_dict, score_i2t_dict):
    logger.info("Computing image2image score.")
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    embedding_type = args.learnable_property
    clip_model, clip_transform = clip.load("ViT-B/32", device=accelerator.device, jit=False)
    clip_model.eval()

    # Update i2i score dict
    if embedding_type == "object":
        i2i_prompt = "A photo of {}".format(args.placeholder_token)  # Or "A photo depicts <*>"
    elif embedding_type == "style":
        i2i_prompt = "A painting in the style of {}".format(
            args.placeholder_token)  # Or "A picture in the style of <*>"
    else:
        raise ValueError("Embedding type should be either 'object' or 'style'")

    i2i_N = args.score_number  # number of random generated images
    i2i_clip_image_dir = os.path.join(args.output_dir, "clip_images")
    os.makedirs(i2i_clip_image_dir, exist_ok=True)

    for n in range(i2i_N):
        image_n = pipeline(i2i_prompt, num_inference_steps=50, guidance_scale=7.5,
                           generator=generator).images[0]
        image_n_path = os.path.join(i2i_clip_image_dir, "{}_{}.png".format(i2i_prompt, n + 1))
        image_n.save(image_n_path)

    i2i_clip_images_path_list = [os.path.join(i2i_clip_image_dir, path) for path in os.listdir(i2i_clip_image_dir)
                                 if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    i2i_original_images_path_list = [os.path.join(args.train_data_dir, path) for path in os.listdir(
        args.train_data_dir) if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    i2i_num_original_images = len(i2i_original_images_path_list)

    i2i_clip_features = extract_all_images(i2i_clip_images_path_list, clip_model, accelerator.device, batch_size=i2i_N,
                                           num_workers=8)
    i2i_original_features = extract_all_images(i2i_original_images_path_list, clip_model, accelerator.device,
                                               batch_size=i2i_num_original_images, num_workers=8)

    # Compute pair-wise Clip-space cosine similarity
    i2i_final_score = get_clip_score(clip_model, i2i_clip_features, i2i_original_features, accelerator.device)
    score_i2i_dict.update({global_step: i2i_final_score.astype(np.float32).tolist()})
    rmtree(i2i_clip_image_dir)

    # Update i2t score dict
    if embedding_type == "object":
        i2t_prompt_list = [
            # background modifications
            "a photo of {}".format(args.placeholder_token),
            "a photo of {} on the beach".format(args.placeholder_token),
            "a photo of {} on the moon".format(args.placeholder_token),
            "a photo of {} on the table".format(args.placeholder_token),
        ]
    elif embedding_type == "style":
        i2t_prompt_list = [
            # object changes
            "a cat in the style of {}".format(args.placeholder_token),
            "an apple in the style of {}".format(args.placeholder_token),
            "a church in the style of {}".format(args.placeholder_token),
            "a waterfall in the style of {}".format(args.placeholder_token),
        ]
    else:
        raise ValueError("Embedding type should be either 'object' or 'style'")

    i2t_N = args.score_number  # number of random generated images
    i2t_clip_image_dir = os.path.join(args.output_dir, "clip_images_temp")
    i2t_total_score = 0

    for i, prompt in enumerate(i2t_prompt_list):
        os.makedirs(i2t_clip_image_dir, exist_ok=True)

        for n in range(i2t_N):
            image_n = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5,
                               generator=generator).images[0]
            image_n_path = os.path.join(i2t_clip_image_dir, "{}_{}.png".format(prompt, n + 1))
            image_n.save(image_n_path)

        i2t_clip_images_path_list = [os.path.join(i2t_clip_image_dir, path) for path in os.listdir(
            i2t_clip_image_dir) if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        i2t_clip_features = extract_all_images(i2t_clip_images_path_list, clip_model, accelerator.device,
                                               batch_size=i2t_N, num_workers=8)

        # get text features
        i2t_text_candidates = [prompt] * i2t_N  # .replace(trained_token, initialization_word)
        text_features = extract_all_captions(i2t_text_candidates, clip_model, accelerator.device, batch_size=i2t_N,
                                             num_workers=8)

        # compute Clip-space cosine similarity
        i2t_once_score = get_clip_score(clip_model, i2t_clip_features, text_features, accelerator.device)
        i2t_total_score += 2.5 * i2t_once_score

        # empty the clip_image_dir
        rmtree(i2t_clip_image_dir)

    # compute and save the final score
    i2t_final_score = i2t_total_score / len(i2t_prompt_list)
    score_i2t_dict.update({global_step: i2t_final_score.astype(np.float32).tolist()})

    return score_i2i_dict, score_i2t_dict


def weight_summation(weight_vector, accelerator, global_step, weight_sum_dict):
    weight_vector = accelerator.unwrap_model(weight_vector).weight
    weight_sum = torch.sum(weight_vector)
    weight_sum_dict.update({global_step: weight_sum.cpu().detach().numpy().astype(np.float32).tolist()})

    return weight_sum_dict


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tq.tqdm(data):
            b = b['image'].to(device)
            b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def extract_all_captions(captions, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tq.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def get_clip_score(model, clip_images, original_images, device, w=1.0):
    if isinstance(clip_images, list):
        # need to extract image features
        clip_images = extract_all_images(clip_images, model, device)
    if isinstance(original_images, list):
        # need to extract image features
        original_images = extract_all_images(original_images, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        clip_images = sklearn.preprocessing.normalize(clip_images, axis=1)
        original_images = sklearn.preprocessing.normalize(original_images, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than'
            'paper results. To exactly replicate paper results, please use numpy version less'
            'than 1.21, e.g., 1.20.3.')
        clip_images = clip_images / np.sqrt(np.sum(clip_images ** 2, axis=1, keepdims=True))
        original_images = original_images / np.sqrt(np.sum(original_images ** 2, axis=1,
                                                           keepdims=True))

    per = w * np.clip(np.dot(clip_images, original_images.T), 0, None)
    return np.mean(per)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument(
        "--second_token", type=str, default=None, help="Second token to accelerate training and convergence."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--dist_type",
        type=str,
        default="Vector_dot",
        choices=["L2", "Cosine_similarity", "Vector_dot"],
        help="Distance to measure the nearest neighbour."
    )
    parser.add_argument(
        "--mask_k",
        type=int,
        default=768,
        help="Number of candidate embeddings."
    )
    parser.add_argument(
        "--lambda_weight",
        type=float,
        default=0.0,
        help="Coefficient of regularization term for w_1 + ... + w_n = 1."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="The directory to keep the pre-trained model file"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--test_score",
        action="store_true",
        help="Whether or not to test the similarity scores"
    )
    parser.add_argument(
        "--score_steps",
        type=int,
        default=10,
        help="Save intermediate similarity scores."
    )
    parser.add_argument(
        "--score_number",
        type=int,
        default=64,
        help="Number of images generated to estimate the clip scores."
    )
    parser.add_argument(
        "--test_weight",
        action="store_true",
        help="Check the summation of the weights."
    )
    parser.add_argument(
        "--weight_sum_steps",
        type=int,
        default=50,
        help="Save intermediate weight summation."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class TextualInversionDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property="object",  # [object, style]
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            placeholder_token="*",
            center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)
                            if file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small \
            if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


class WeightVector(nn.Module):
    def __init__(self, args):
        super().__init__()
        init_weight = torch.zeros([args.mask_k, 1])
        init_weight[0] = 1.0
        weight_param = nn.Parameter(init_weight, requires_grad=True)
        self.register_parameter("weight", weight_param)

    '''
    def forward(self, candidate_embedding_matrix):
        updated_embedding = torch.mm(torch.t(self.weight).to(candidate_embedding_matrix.device),
                                     candidate_embedding_matrix).squeeze(0)
        return updated_embedding
    '''

    def forward(self, text_encoder, input_ids, candidate_embedding_matrix, placeholder_token_id):
        # TODO: Freeze the parameters in text_encoder.text_model.embeddings.token_embedding.weight
        # TODO: When calling the placeholder_token_id, using the weight_vector instead
        # Perform in CLIPTextTransformer
        output_attentions = text_encoder.text_model.config.output_attentions
        output_hidden_states = text_encoder.text_model.config.output_hidden_states
        return_dict = text_encoder.text_model.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # Begin of CLIPTextEmbeddings
        # hidden_states = text_encoder.text_model.embeddings(input_ids=input_ids)
        seq_length = input_ids.shape[-1]
        position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]

        # TODO: Change here
        # The shape should be [1, 77, 768]
        # inputs_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)

        # Begin modification
        inputs_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids).detach()
        input_ids_numpy = input_ids.squeeze(0).cpu().numpy()
        placeholder_token_index = np.argwhere(input_ids_numpy == placeholder_token_id)[0][0]
        inputs_embeds[:, placeholder_token_index, :] = torch.mm(
            torch.t(self.weight).to(candidate_embedding_matrix.device), candidate_embedding_matrix).squeeze(0)
        # End modification

        position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings
        # End of CLIPTextEmbeddings

        bsz, seq_len = input_shape
        causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len,
                                                                                     hidden_states.dtype).to(
            hidden_states.device)

        # Begin of CLIPEncoder
        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        '''
        output_attentions = output_attentions if output_attentions is not None else text_encoder.text_model.encoder.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else text_encoder.text_model.encoder.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else text_encoder.text_model.encoder.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, encoder_layer in enumerate(text_encoder.text_model.encoder.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if text_encoder.text_model.encoder.gradient_checkpointing and text_encoder.text_model.encoder.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    None,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            encoder_outputs = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states,
                                              attentions=all_attentions)
        '''
        # End of CLIPEncoder

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        if not return_dict:
            updated_embedding = (last_hidden_state, pooled_output) + encoder_outputs[1:]
        else:
            updated_embedding = BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        return updated_embedding


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Convert the second_token if it exists
    if args.second_token is not None:
        token_ids_second = tokenizer.encode(args.second_token, add_special_tokens=False)
        if len(token_ids_second) > 1:
            raise ValueError("The second token must be a single token.")
        second_token_id = token_ids_second[0]
    else:
        second_token_id = None

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    # Get the original vocabulary embedding
    vocab_embedding = token_embeds[:-1, :]
    vocab_num = len(tokenizer) - 1

    # Select M nearest neighbours
    initialization_embedding = token_embeds[initializer_token_id]
    vocab_ids = None
    candidate_embedding_matrix = None

    if args.second_token is None:
        if args.dist_type == "Vector_dot":
            vocab_dist = torch.mm(vocab_embedding, initialization_embedding.unsqueeze(1)).squeeze(1)
            _, vocab_ids = torch.topk(vocab_dist, args.mask_k, 0, True)
            candidate_embedding_matrix = token_embeds[vocab_ids]  # [args.mask_k, 768]
        elif args.dist_type == "Cosine_similarity":
            embeds_matrix = initialization_embedding.unsqueeze(0).expand(vocab_num, initialization_embedding.shape[0])
            vocab_dist = torch.cosine_similarity(embeds_matrix, vocab_embedding, 0)
            _, vocab_ids = torch.topk(vocab_dist, args.mask_k, 0, True)
            candidate_embedding_matrix = token_embeds[vocab_ids]  # [args.mask_k, 768]
        elif args.dist_type == "L2":
            embeds_matrix = initialization_embedding.unsqueeze(0).expand(vocab_num, initialization_embedding.shape[0])
            residual_matrix = embeds_matrix - vocab_embedding
            vocab_dist = torch.norm(residual_matrix, 2, 1)
            _, vocab_ids = torch.topk(vocab_dist, args.mask_k, 0, False)
            candidate_embedding_matrix = token_embeds[vocab_ids]  # [args.mask_k, 768]
    else:
        second_embedding = token_embeds[second_token_id]
        if args.dist_type == "Vector_dot":
            vocab_dist_first = torch.mm(vocab_embedding, initialization_embedding.unsqueeze(1)).squeeze(1)
            _, vocab_ids_first = torch.topk(vocab_dist_first, args.mask_k - 1, 0, True)
            if second_token_id in vocab_ids_first:
                raise ValueError("The candidates already contain the second token. Please disable --second_token!")
            else:
                candidate_embedding_matrix_first = token_embeds[vocab_ids_first]
                second_token_id_tensor = torch.tensor([second_token_id], dtype=torch.int64)
                vocab_ids = torch.cat((vocab_ids_first, second_token_id_tensor), 0)
                candidate_embedding_matrix = torch.cat(
                    (candidate_embedding_matrix_first, second_embedding.unsqueeze(0)), 0)
        elif args.dist_type == "Cosine_similarity":
            embeds_matrix = initialization_embedding.unsqueeze(0).expand(vocab_num, initialization_embedding.shape[0])
            vocab_dist_first = torch.cosine_similarity(embeds_matrix, vocab_embedding, 0)
            _, vocab_ids_first = torch.topk(vocab_dist_first, args.mask_k - 1, 0, True)
            if second_token_id in vocab_ids_first:
                raise ValueError("The candidates already contain the second token. Please disable --second_token!")
            else:
                candidate_embedding_matrix_first = token_embeds[vocab_ids_first]
                second_token_id_tensor = torch.tensor([second_token_id], dtype=torch.int64)
                vocab_ids = torch.cat((vocab_ids_first, second_token_id_tensor), 0)
                candidate_embedding_matrix = torch.cat(
                    (candidate_embedding_matrix_first, second_embedding.unsqueeze(0)), 0)
        elif args.dist_type == "L2":
            raise NotImplementedError

    '''
    # Initialize weights and embedding
    weight_vector = torch.zeros([args.mask_k, 1])
    weight_vector[0] = 1.0  # initialize as TI
    # register weight_vector and candidate_embedding_matrix
    weight_vector = torch.nn.Parameter(weight_vector)
    '''
    weight_vector = WeightVector(args)
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        weight_vector.parameters(),  # only optimize the weight_vector
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler, weight_vector, candidate_embedding_matrix, vocab_ids = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, weight_vector, candidate_embedding_matrix, vocab_ids)

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("linear_regression", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    # TEST code!!!
    # orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    # orig_weights = accelerator.unwrap_model(weight_vector).weight.clone()

    if args.test_score:
        score_i2i_dict = {}
        score_i2t_dict = {}
    else:
        score_i2i_dict = None
        score_i2t_dict = None

    if args.test_weight:
        weight_sum_dict = {}
    else:
        weight_sum_dict = None

    for epoch in range(first_epoch, args.num_train_epochs):
        weight_vector.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(weight_vector):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # text_encoder.module.get_input_embeddings().weight.requires_grad_(False)
                '''
                text_encoder.text_model.embeddings.token_embedding.weight[placeholder_token_id] = weight_vector.forward(
                    candidate_embedding_matrix)
                # text_encoder.module.get_input_embeddings().weight.requires_grad_(True)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                '''
                encoder_hidden_states = weight_vector(text_encoder, batch["input_ids"], candidate_embedding_matrix,
                                                      placeholder_token_id)[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Add regularization term
                if args.lambda_weight != 0.0:
                    weight_sum = torch.sum(weight_vector.weight)
                    loss_reg = torch.abs(
                        torch.ones(1, dtype=weight_sum.dtype, device=weight_sum.device)[0] - weight_sum)
                    loss = loss + args.lambda_weight * loss_reg

                accelerator.backward(loss)  # retain_graph=True

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Keep the embeddings the same
                # accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data = orig_embeds_params
                '''
                index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                '''
                # Update the learned embedding
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        placeholder_token_id] = torch.mm(torch.t(accelerator.unwrap_model(weight_vector).weight).to(
                        accelerator.unwrap_model(candidate_embedding_matrix).device),
                        accelerator.unwrap_model(candidate_embedding_matrix)).squeeze(
                        0)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # TEST code!!!
                '''
                now_embedding = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data
                print(f"Embeddings are the same: {torch.equal(now_embedding, orig_embeds_params)}")
                now_weight = accelerator.unwrap_model(weight_vector).weight
                print(f"Weights are the same: {torch.equal(now_weight, orig_weights)}")
                if global_step % 10 == 0:
                    raise ValueError
                '''

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, weight_vector, candidate_embedding_matrix,
                                  vocab_ids, accelerator, args, save_path)
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step)
                if args.test_score and global_step % args.score_steps == 0:
                    score_i2i_dict, score_i2t_dict = score_computation(text_encoder, tokenizer, unet, vae, args,
                                                                       accelerator, weight_dtype, global_step,
                                                                       score_i2i_dict, score_i2t_dict)
                if args.test_weight and global_step % args.weight_sum_steps == 0:
                    weight_sum_dict = weight_summation(weight_vector, accelerator, global_step, weight_sum_dict)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            os.makedirs(args.model_dir, exist_ok=True)
            pipeline.save_pretrained(args.model_dir)
        # Save the newly trained embeddings
        if not args.test_score:
            save_path = os.path.join(args.output_dir, "learned_embeds.bin")
            save_progress(text_encoder, placeholder_token_id, weight_vector, candidate_embedding_matrix, vocab_ids,
                          accelerator, args, save_path)

        # Save the type of embedding
        type_path = f"{args.output_dir}/type_of_concept.txt"
        with open(type_path, "w") as f:
            f.write(f"{args.learnable_property}")

        # Save the initialization word
        init_word_path = f"{args.output_dir}/initialization_word.txt"
        with open(init_word_path, "w") as f:
            f.write(f"{args.initializer_token}")

        # Write score dict to txt
        if args.test_score:
            test_score_path = f"{args.output_dir}/test_score"
            os.makedirs(test_score_path, exist_ok=True)
            test_i2i_path = f"{test_score_path}/i2i_score.txt"
            test_i2t_path = f"{test_score_path}/i2t_score.txt"
            with open(test_i2i_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(score_i2i_dict))
            with open(test_i2t_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(score_i2t_dict))

        # Write weight sum dict to txt
        if args.test_weight:
            test_weight_path = f"{args.output_dir}/test_weight"
            os.makedirs(test_weight_path, exist_ok=True)
            test_weight_sum_path = f"{test_weight_path}/weight_summation.txt"
            with open(test_weight_sum_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(weight_sum_dict))

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
