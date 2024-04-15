import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import time
import importlib
import argparse
# from tqdm import tqdm
from tqdm.auto import tqdm
from dataset_catalog import DatasetCatalog


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser.add_argument(
    #     "--pretrained_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
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
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    total_steps = 256
    batch_size = 1
    num_inference_steps = 1
    num_images_per_prompt = 1
    guidance_scale = 7.5
    device = 'cuda'
    height = 256
    width = 256
    vae_scale_factor = 8

    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # pipe.safety_checker = None
    # pipe = pipe.to(device)

    vae = AutoencoderKL(**pipe.vae.config)
    vae.load_state_dict(pipe.vae.state_dict())
    vae = vae.to(device)
    vae.eval()
    text_encoder = CLIPTextModel(pipe.text_encoder.config)
    text_encoder.load_state_dict(pipe.text_encoder.state_dict())
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    tokenizer = pipe.tokenizer #CLIPTokenizer()
    unet = UNet2DConditionModel(**pipe.unet.config)
    unet.load_state_dict(pipe.unet.state_dict())
    unet = unet.to(device)
    scheduler = PNDMScheduler(**pipe.scheduler.config)
    feature_extractor = pipe.feature_extractor

    def get_dataset():
        """Instantiate the dataset object."""
        Catalog = DatasetCatalog(root_path='../../data')

        dataset_dict = getattr(Catalog, 'ImageNetCDataset') # args.dataset_name

        module, cls = dataset_dict['target'].rsplit(".", 1)
        params = dataset_dict['train_params']
        dataset = getattr(importlib.import_module(module, package=None), cls)

        return dataset(**params)

    dataset = get_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2, #args.dataloader_num_workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )

    # pipe.text_encoder.requires_grad_(False)
    # pipe.unet.requires_grad_(True)
    # # pipe.unet.requires_grad_(False)
    # pipe.vae.requires_grad_(False)
    # pipe.unet.set_attn_processor(AttnProcessor2_0())
    # pipe.scheduler.set_timesteps(50, device=device)

    optimizer = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    # img_tensor = pipe('an image of an airplane', num_inference_steps=num_inference_steps, output_type='pt')
    # img = pipe('an image of an airplane', num_inference_steps=num_inference_steps)['images'][0]
    # img.save('airplane_5step.png')

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    model.to(device)
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    # pipe.unet.train()
    for idx, batch in enumerate(dataloader):
        # image = pipe('an image of {}'.format(batch['class_descriptions']), num_inference_steps=num_inference_steps, output_type='pt', safety_checker=None)
        # image = preprocess(image.images[0]).unsqueeze(0)
        # logits = model(image)
        # loss_cls = ce_loss(logits.cuda(), batch['class_idx'].long().cuda())
        # print('classification: {:.4f}'.format(loss_cls.item()))
        #
        # loss_cls.backward()
        # optimizer.step()
        # # lr_scheduler.step()
        # optimizer.zero_grad()

        # vae encoding
        # latent = vae.encode(batch['image_disc'])
        # loss = torch.norm(latent.latent_dist.mean)
        # loss.backward()
        # Encoding prompt of class names
        text_inputs = tokenizer(['An image of {}'.format(x) for x in batch['class_descriptions']], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=None,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        # loss = torch.sum(prompt_embeds)
        # loss.backward()
        # 4. Prepare timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        latents = randn_tensor(shape, generator=None, device=device, dtype=prompt_embeds.dtype)
        latents = latents * scheduler.init_noise_sigma

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        _num_timesteps = len(timesteps)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                # latent_model_input = torch.nn.Parameter(latent_model_input)
                # latent_model_input.requires_grad_(False)
                # predict the noise residual
                noise_pred, t1, t2 = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )
                noise_pred.sum().backward()

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                loss = torch.norm(noise_pred)
                loss.backward()
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        loss = torch.norm(image)
        loss.backward()
        do_denormalize = [True] * image.shape[0]
        image = torch.stack([(image[i] / 2 + 0.5).clamp(0, 1) if do_denormalize[i] else image[i] for i in range(image.shape[0])])

        # for device_step in tqdm(range(total_steps)):
        # _t = 0
        # for t in tqdm(timesteps):
        #     # Noise
        #     # noise = torch.randn_like(latents)
        #     # timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        #     # timesteps = timesteps.long()
        #
        #     # noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        #     noisy_latents = pipe.scheduler.scale_model_input(latents, t)
        #
        #     # Conditioning
        #     encoder_hidden_states = [pipe.text_encoder(torch.asarray(input_id).to(device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]
        #     encoder_hidden_states = torch.stack(tuple(encoder_hidden_states))
        #
        #     # Forward pass
        #     with torch.autocast(device, enabled=True):
        #         model_pred = pipe.unet(noisy_latents, t, encoder_hidden_states).sample
        #         # latents = torch.stack([pipe.scheduler.step(model_pred[i], timesteps[i].item(), noisy_latents[i], return_dict=False)[0] for i in range(batch_size)])
        #         latents = pipe.scheduler.step(model_pred, t.item(), noisy_latents, return_dict=False)[0]
        #         image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        #         if _t % 10 == 0:
        #             vis_image = pipe.image_processor.postprocess(image.detach(), output_type='pil', do_denormalize=[True] * batch_size)
        #         image = pipe.image_processor.postprocess(image, output_type='pt', do_denormalize=[True] * batch_size)
        #         image = preprocess(image)
        #         prediction = model(image).softmax(1)
        #         loss_cls = ce_loss(prediction, batch['class_idx'].to(device))
        #         # Loss
        #         # loss_diff = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        #         # loss_diff = loss_diff / total_steps
        #
        #         # print('diffusion: {:.4f}, classification: {:.4f}'.format(loss_diff, loss_cls))
        #         print('classification: {:.4f}'.format(loss_cls))
        #         loss = loss_cls
        #
        #         if _t % 10 == 0:
        #             j = 0
        #             gt_classes = [dataset.class_names[c.item()] for c in batch['class_idx']]
        #             pred_classes = [dataset.class_names[c.item()] for c in prediction.argmax(dim=1)]
        #             for v_img, gt, pred in zip(vis_image, gt_classes, pred_classes):
        #                 v_img.save('./{}th_batch_{}th_timestep_gt_{}_pred_{}_{}.png'.format(idx, t, gt, pred, j))
        #                 j += 1
        #         _t += 1

            # Backward pass
            # scaler.scale(loss).backward()

    end_time = time.time()
    print()
    print("Done")
    print("Time:", end_time - start_time)
    print("Average time per step:", (end_time - start_time) / total_steps)

    pipe.unet.requires_grad_(False)
    image_after_adapt = pipe(prompt).images[0]
    image_after_adapt.save('tmp1_after_adapt.png')


if __name__ == '__main__':
    main()