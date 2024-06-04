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
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import cv2

# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor
from PIL import Image
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset, Dataset
from torch.utils.data import Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers import StableDiffusionGLIGENPipeline

from detectron2.data import MetadataCatalog, DatasetCatalog, get_detection_dataset_dicts
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

class FromDetectron2Dataset(Dataset):
    def __init__(self, dataset_name, split="train", preprocess=None):
        self.dataset_name = "_".join(dataset_name.split("_")[:-1] + [split, dataset_name.split("_")[-1]])
        self.data = get_detection_dataset_dicts(self.dataset_name, filter_empty=True)
        self.class_names = MetadataCatalog.get(self.dataset_name).thing_classes
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Load the image from the file path and convert it to a tensor
        image = Image.open(item['file_name'])
        height, width = image.height, image.width
        crop_size = min(height, width)
        x_offset = (width - crop_size) // 2
        phrases = []
        boxes = []
        for i in item['annotations']:
            x1, y1, x2, y2 = i['bbox'] # XYXY
            # Adjust the coordinates based on the offsets
            x1_new = (x1 - x_offset) / crop_size
            x2_new = (x2 - x_offset) / crop_size
            y1_new = y1 / crop_size
            y2_new = y2 / crop_size
            if x1_new < 1 and x2_new > 0 and y1_new < 1 and y2_new > 0:
                x1_new = max(0, x1_new)
                x2_new = min(1, x2_new)
                y1_new = max(0, y1_new)
                y2_new = min(1, y2_new)
                phrases.append(self.class_names[i['category_id']])
                boxes.append([x1_new, y1_new, x2_new, y2_new])

        unique_classes = list(set(phrases))
        text = "An image with {}".format(", ".join(unique_classes))
        item = {'image': image, 'text': text, 'phrases': phrases, 'boxes': boxes}
        if self.preprocess:
            self.preprocess(item)
        item['class_tokens'] = [(c, c_idx * 2 + 4) for c_idx, c in enumerate(unique_classes)]
        return item


def postprocess_mask(mask, image_pixels=None):
    img_h, img_w = image_pixels.shape[1:] if image_pixels is not None else (512, 512)
    h, w = int(mask.shape[-1] ** 0.5), int(mask.shape[-1] ** 0.5)
    resized_mask = mask.detach().cpu().numpy()
    resized_mask = resized_mask.squeeze().mean(axis=0)
    resized_mask = (resized_mask - resized_mask.min()) / (resized_mask.max() - resized_mask.min())
    resized_mask = resized_mask * 255
    resized_mask = resized_mask.reshape(h, w)
    resized_mask = cv2.resize(resized_mask, (img_h, img_w))
    # resized_mask = cv2.applyColorMap(resized_mask.astype('uint8'), cv2.COLORMAP_JET)
    return Image.fromarray(resized_mask.astype('uint8'))

def visualize_mask_on_image(images, masks, tracker, wandb, validation_prompt):
    # Go through each mask
    # visualize attention masks on the image
    for b, image in enumerate(images):
        for i, mask in enumerate(masks):
            for t in range(mask.shape[-1]):
                overlay = np.zeros_like(image)
                h, w = int(mask.shape[-2] ** 0.5), int(mask.shape[-1] ** 0.5)
                attn_mask = mask[b, :, :, t].reshape(-1, h, w)
                for j in range(attn_mask.shape[0]):
                    cur_mask = ((attn_mask[j] - attn_mask[j].min()) * (1 / (attn_mask[j].max() - attn_mask[j].min()) * 255)).cpu().numpy().astype('uint8')
                    cur_mask = cv2.resize(cur_mask, (overlay.shape[1], overlay.shape[0]))
                    colored_mask = cv2.applyColorMap(cur_mask, cv2.COLORMAP_JET)
                    alpha_channel = cur_mask.copy()
                    colored_mask_with_alpha = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2BGRA)
                    colored_mask_with_alpha[:, :, 3] = alpha_channel
                    overlay += cur_mask
                overlay = (overlay / attn_mask.shape[0]).astype('uint8')
                blended = cv2.addWeighted(np.array(image), 0.7, overlay, 0.3, 0)
                blended_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
                tracker.log(
                    {
                        f"attention_mask_{i}th_hw{h}": wandb.Image(blended_pil, caption=f"token{t}: {validation_prompt}"
                        )
                    }
                )


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
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
        help=("Max number of checkpoints to store."),
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.pretrained_model_name_or_path)
    # pipe = pipe.to("cuda")
    # pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, revision="fp16")
    #
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    # )
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    # )
    # freeze parameters of models to save more memory
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in pipe.unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    #unet.to(accelerator.device, dtype=weight_dtype)
    pipe.unet.to(accelerator.device)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    pipe.unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(pipe.unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipe.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, pipe.unet.parameters())

    if args.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, caption_column="text"):
        inputs = pipe.tokenizer(
            examples[caption_column], max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples, image_column="image"):
        image = examples[image_column].convert("RGB")
        examples["pixel_values"] = train_transforms(image)
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = dataset["train"].with_transform(preprocess_train)

    train_dataset = FromDetectron2Dataset(args.dataset_name, split="train", preprocess=preprocess_train)
    # teat_dataset = FromDetectron2Dataset(args.dataset_name, split="test")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        class_tokens = [example["class_tokens"] for example in examples]
        captions = [example["text"] for example in examples]
        phrases = [example["phrases"] for example in examples]
        boxes = [example["boxes"] for example in examples]
        return {"pixel_values": pixel_values, "input_ids": input_ids, "class_tokens": class_tokens, "captions": captions, "phrases": phrases, "boxes": boxes}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    pipe.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                pipe = pipe.to(accelerator.device)
                pipe.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                prompt_list = ["", "A photo of ", "An image of ", "An image with "]

                boxes = []
                for _ in range(3):
                    x, y, w, h = random.random(), random.random(), random.random() * 0.2 + 0.1, random.random() * 0.2 + 0.1
                    box = [max(min(x, 0.9), 0.01), max(min(y, 0.9), 0.01), min(x + w, 0.99), min(y + h, 0.99)]
                    boxes.append(box)
                phrases = random.choices(train_dataset.class_names, k=3)

                masks = []
                with torch.cuda.amp.autocast():
                    val_img = pipe(prompt="A photo", gligen_phrases = phrases, gligen_boxes = boxes, num_inference_steps=30, generator=generator, cross_attention_kwargs={'masks': masks}).images[0]
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [wandb.Image(val_img, caption=f"{epoch}: {'_'.join(phrases)}")]
                            }
                        )
                torch.cuda.empty_cache()

        pipe.unet.train()
        train_loss = 0.0
        max_objs = 30
        for step, batch in enumerate(train_dataloader):
            with ((accelerator.accumulate(pipe.unet))):
                # break down the pipe call function to train only adaptor added unet
                # Convert images to latent space
                latents = pipe.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]

                device = pipe._execution_device

                # Sample a random timestep for each image
                # StableDiffusionGLIGENPipeline use scheduler, not noise_scheduler
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = pipe.text_encoder(batch["input_ids"], return_dict=False)[0]

                tokenizer_inputs = pipe.tokenizer(batch["phrases"][0], padding=True, return_tensors="pt").to(device)
                _text_embeddings = pipe.text_encoder(**tokenizer_inputs).pooler_output
                text_embeddings = torch.zeros(
                    max_objs, pipe.unet.config.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
                )

                n_objs = len(batch["boxes"][0])
                text_embeddings[:n_objs] = _text_embeddings

                boxes = torch.zeros(max_objs, 4, device=device, dtype=pipe.text_encoder.dtype)
                boxes[:n_objs] = torch.tensor(batch["boxes"][0])

                masks = torch.zeros(max_objs, device=device, dtype=pipe.text_encoder.dtype)
                masks[:n_objs] = 1

                boxes = boxes.unsqueeze(0).expand(bsz, -1, -1)
                text_embeddings = text_embeddings.unsqueeze(0).expand(bsz, -1, -1)
                masks = masks.unsqueeze(0).expand(bsz, -1)
                cross_attention_kwargs = {}
                cross_attention_kwargs["gligen"] = {"boxes": boxes, "positive_embeddings": text_embeddings,
                                                    "masks": masks}

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    pipe.scheduler.register_to_config(prediction_type=args.prediction_type)

                if pipe.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif pipe.scheduler.config.prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {pipe.scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(pipe.scheduler, timesteps)
                    if pipe.scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # import pdb; pdb.set_trace()
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # unwrapped_unet = unwrap_model(unet)
                        # unet_lora_state_dict = convert_state_dict_to_diffusers(
                        #     get_peft_model_state_dict(unwrapped_unet)
                        # )
                        #
                        # StableDiffusionPipeline.save_lora_weights(
                        #     save_directory=save_path,
                        #     unet_lora_layers=unet_lora_state_dict,
                        #     safe_serialization=True,
                        # )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    #
    # # Save the lora layers
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = pipe.unet.to(torch.float32)
    #
    #     unwrapped_unet = unwrap_model(unet)
    #     unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    #     StableDiffusionPipeline.save_lora_weights(
    #         save_directory=args.output_dir,
    #         unet_lora_layers=unet_lora_state_dict,
    #         safe_serialization=True,
    #     )
    #
    #     # Final inference
    #     # Load previous pipeline
    #     if args.validation_prompt is not None:
    #         pipeline = DiffusionPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path,
    #             revision=args.revision,
    #             variant=args.variant,
    #             torch_dtype=weight_dtype,
    #         )
    #         pipeline = pipeline.to(accelerator.device)
    #
    #         # load attention processors
    #         pipeline.load_lora_weights(args.output_dir)
    #
    #         # run inference
    #         generator = torch.Generator(device=accelerator.device)
    #         if args.seed is not None:
    #             generator = generator.manual_seed(args.seed)
    #         images = []
    #         with torch.cuda.amp.autocast():
    #             for _ in range(args.num_validation_images):
    #                 images.append(
    #                     pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
    #                 )
    #
    #         for tracker in accelerator.trackers:
    #             if len(images) != 0:
    #                 if tracker.name == "tensorboard":
    #                     np_images = np.stack([np.asarray(img) for img in images])
    #                     tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #                 if tracker.name == "wandb":
    #                     tracker.log(
    #                         {
    #                             "test": [
    #                                 wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                                 for i, image in enumerate(images)
    #                             ]
    #                         }
    #                     )

    accelerator.end_training()


if __name__ == "__main__":
    main()
