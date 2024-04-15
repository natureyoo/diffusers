import argparse
import logging
import os
import sys
import random
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from diffusers import StableDiffusionGLIGENTextImagePipeline
import wandb
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

# Initialize wandb
wandb.init(project='Gligen', entity='natureyoo')

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--model-dir", type=str, default="./VOC_baseline/model_final.pth", help="perform evaluation only")
    parser.add_argument("--feats-path", type=str, default="", help="perform evaluation only")
    parser.add_argument("--loss-type", type=str, default="feature", help="perform evaluation only")
    parser.add_argument("--guidance_step", type=int, default=1000, help="timestep guidance")
    parser.add_argument("--guidance_weight", type=float, default=1.0, help="guidance weight")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def compute_detector_loss(detector, image, feats_bank, phrases, boxes, cls_idxes, loss_type="feature"):
    target_classes = [cls_idxes[p] for p in phrases]
    # Compute detector loss
    # new_imgs.append(images[0].resize((1024, 1024)))
    new_image = torch.cat([image[0], image[0]], dim=2) * 255
    # new_image.requires_grad = True
    new_h, new_w = new_image.shape[1], new_image.shape[2]
    synth_instances = Instances((new_h, new_w))

    # add boxes to gt instances
    new_boxes = boxes + [[box[0] + 1, box[1], box[2] + 1, box[3]] for box in boxes]
    new_boxes = torch.FloatTensor(new_boxes) * new_h
    synth_instances.gt_boxes = Boxes(new_boxes)
    synth_instances.gt_classes = torch.LongTensor(target_classes * 2)

    # add proposal boxes
    new_inputs = [{"image": new_image, "instances": synth_instances}]
    box_features, predictions, gt_classes = detector.extract_features(new_inputs)

    loss_detector = 0
    if loss_type == "feature":
        for c in torch.unique(gt_classes):
            if feats_bank[c.item()].shape[0] > 0:
                loss_detector += torch.dist(feats_bank[c.item()].mean(dim=0), box_features[gt_classes == c].mean(dim=[2,3]).mean(dim=0))
    else:
        # detector loss
        loss_detector = torch.nn.CrossEntropyLoss()(predictions[0], gt_classes)
    # feature loss
    return loss_detector


def evaluate_detection(detector, img, instances, stat_by_cls):
    # add proposal boxes
    # rescale it to the original image size
    instances.gt_boxes.scale(img.shape[2] / instances.image_size[1], img.shape[1] / instances.image_size[0])
    instances._image_size = (img.shape[1], img.shape[2])
    new_inputs = [{"image": img, "instances": instances}]
    box_features, predictions, gt_classes = detector.extract_features(new_inputs)
    probs = torch.nn.Softmax(dim=1)(predictions[0])
    conf, pred_class = probs.max(dim=1)
    for _c in torch.unique(gt_classes):
        c = int(_c.item())
        stat_by_cls['prob'][c] = torch.cat([stat_by_cls['prob'][c], probs[gt_classes == c][:, [c, -1]].detach().cpu()], dim=0)  # first column is gt prob and second column is background prob
        stat_by_cls['correct'][c] += (pred_class[gt_classes == c] == c).sum().item()
        stat_by_cls['total'][c] += (gt_classes == c).sum().item()



def get_noised_feature(pipe, detector, dataloader, classes):
    features = {}
    # Compute detector loss
    for idx, inputs in enumerate(dataloader):
        pass

def generate_image(cfg, pipe, detector, data_loader, class_names, feats_bank=None, save_dir='./outputs'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Annotations"), exist_ok=True)

    # random.shuffle(file_list)
    novel_classes = ['a {}'.format(c) for c in class_names]
    cls_idxes = {'a {}'.format(c): i for i, c in enumerate(class_names)}
    for idx, inputs in enumerate(data_loader):
        # input_image = Image.open(
        #     os.path.join("/data/Dataset/cityscapes_vocstyle/VOC2007/JPEGImages", "{}.jpg".format(f)))
        # img1 = input_image.crop((0, 0, 1024, 1024))
        # img2 = input_image.crop((1024, 0, 2048, 1024))
        f_name = inputs[0]['image_id']
        img1, img2 = inputs[0]['image'][:, :, :600], inputs[0]['image'][:, :, 600:]

        # insert three novel class objects
        new_imgs = []
        new_boxes = []
        new_classes = []
        for idx, img in enumerate([img1]):
            img = to_pil_image(img)
            boxes = []
            for _ in range(3):
                x, y, w, h = random.random(), random.random(), random.random() * 0.2 + 0.1, random.random() * 0.2 + 0.1
                box = [max(min(x, 0.9), 0.01), max(min(y, 0.9), 0.01), min(x + w, 0.99), min(y + h, 0.99)]
                boxes.append(box)
            phrases = random.choices(novel_classes, k=3)
            prompt = " and ".join(phrases) + " in the street"

            latents, has_nsfw_concept, inpaint_mask = pipe(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_inpaint_image=img,
                # gligen_inpaint_image=None,
                gligen_boxes=boxes,
                # gligen_images=gligen_images,
                gligen_scheduled_sampling_beta=1,
                output_type="latent",
                num_inference_steps=50,
                vis_period=1,
                wandb=wandb,
                file_name="{}-{}".format(f_name, idx)
            )
            # decode latent to images
            latents.requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=0.00001)  # Use an appropriate learning rate

            for step in range(1):
                optimizer.zero_grad()

                # Generate image from current latents
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                if has_nsfw_concept is None:
                    do_denormalize = [True] * image.shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

                image = pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)

                target_classes = [cls_idxes[p] for p in phrases] * 2
                loss = compute_detector_loss(detector, image, target_classes, boxes)
                loss.backward()  # Maximize the classification accuracy (minimize negative loss)
                # torch.nn.utils.clip_grad_norm_(latents, 1)
                latents = latents - 1 * latents.grad * (1 - inpaint_mask)
                # optimizer.step()

            # visualize output
            with torch.no_grad():
                image = pipe.vae.decode(latents.detach() / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
                wandb.log({"{}-{}".format(f_name, idx): [wandb.Image(image[0], caption=f"after optimize")]})
            print('')


def generate_image_with_detector_guidance(cfg, pipe, detector, data_loader, class_names, feats_bank=None, loss_type="feature", guidance_step=1000, guidance_weight=1.0):
    save_dir = cfg.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Annotations"), exist_ok=True)

    stat_by_cls_synth = {"prob": [torch.Tensor([]) for _ in class_names], "correct": [0 for _ in class_names], "total": [0 for _ in class_names]}
    stat_by_cls_real = {"prob": [torch.Tensor([]) for _ in class_names], "correct": [0 for _ in class_names],
                         "total": [0 for _ in class_names]}

    novel_classes = ['a {}'.format(c) for c in class_names]
    cls_idxes = {'a {}'.format(c): i for i, c in enumerate(class_names)}
    for idx, inputs in enumerate(data_loader):
        if idx > 300:
            break
        f_name = inputs[0]['image_id']
        img1, img2 = inputs[0]['image'][:, :, :600], inputs[0]['image'][:, :, 600:]

        # insert three novel class objects
        new_imgs = []
        new_boxes = []
        new_classes = []
        for idx, img in enumerate([img1, img2]):
            img = to_pil_image(img)
            boxes = []
            for _ in range(3):
                x, y, w, h = random.random(), random.random(), random.random() * 0.2 + 0.1, random.random() * 0.2 + 0.1
                box = [max(min(x, 0.9), 0.01), max(min(y, 0.9), 0.01), min(x + w, 0.99), min(y + h, 0.99)]
                boxes.append(box)
            phrases = random.choices(novel_classes, k=3)
            prompt = " and ".join(phrases) + " in the street"

            images = pipe.generate_with_detector_guidance(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_inpaint_image=img,
                # gligen_inpaint_image=None,
                gligen_boxes=boxes,
                # gligen_images=gligen_images,
                gligen_scheduled_sampling_beta=1,
                output_type="pil",
                num_inference_steps=50,
                vis_period=1,
                wandb=wandb,
                file_name="{}-{}".format(f_name, idx),
                detector=detector,
                feats_bank=feats_bank,
                compute_loss=compute_detector_loss,
                cls_idxes=cls_idxes,
                loss_type=loss_type,
                guidance_step=guidance_step,
                guidance_weight=guidance_weight
            ).images

            new_imgs.append(images[0].resize((1024, 1024)))
            new_boxes += boxes if idx == 0 else [[b[0] + 1, b[1], b[2] + 1, b[3]] for b in boxes]
            new_classes += phrases
        new_img = Image.new('RGB', (2048, 1024))
        new_img.paste(new_imgs[0], (0, 0))
        new_img.paste(new_imgs[1], (1024, 0))
        new_img.save(os.path.join(save_dir, "JPEGImages/{}.jpg".format(f_name)))
        with open(os.path.join(save_dir, "Annotations/{}.txt".format(f_name)), "w") as ann:
            for cls, box in zip(new_classes, new_boxes):
                x1 = int(box[0] * 1024)
                y1 = int(box[1] * 1024)
                x2 = int(box[2] * 1024)
                y2 = int(box[3] * 1024)
                ann.write("{},{},{},{},{}\n".format(cls, x1, y1, x2, y2))

        # evaluate synthetic objects with detector
        w, h = new_img.size
        synth_instances = Instances((h, w))
        # Box!!!
        synth_instances.gt_boxes = Boxes(torch.Tensor([[val * h for val in box] for box in new_boxes]))
        synth_instances.gt_classes = torch.Tensor([cls_idxes[c] for c in new_classes])
        evaluate_detection(detector, pil_to_tensor(new_img), synth_instances, stat_by_cls_synth)
        #
        # # evaluate real objects with detector
        evaluate_detection(detector, pil_to_tensor(new_img), inputs[0]['instances'], stat_by_cls_real)

    import matplotlib.pyplot as plt
    import numpy as np
    # evaluate synthetic objects
    # thresholds = np.arange(0, 1.0, 0.01)
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#123456', '#abcdef']
    # for type_name, stat in zip(["Synthetic", "Real"], [stat_by_cls_synth, stat_by_cls_real]):
    #     fig, ax = plt.subplots()
    #     for c_idx, c_name in enumerate(class_names):
    #         probs = stat['prob'][c_idx].detach().cpu().numpy()
    #         accuracies = []
    #         for p in thresholds:
    #             acc = (probs >= p).sum() / len(probs)
    #             accuracies.append(acc)
    #         ax.plot(thresholds, accuracies, label=c_name, color=colors[c_idx])
    #     ax.set_xlabel('Probability Threshold')
    #     ax.set_ylabel('Accuracy')
    #     ax.legend()
    #
    #     wandb.log({"Accuracy Plot for {}".format(type_name): wandb.Image(plt)})
    #     plt.close()
    for c_idx, c_name in enumerate(class_names):
        probs_synth = stat_by_cls_synth['prob'][c_idx].numpy()
        probs_real = stat_by_cls_real['prob'][c_idx].numpy()
        fig, ax = plt.subplots()
        # binning probability into 10 bins
        bins = np.linspace(0, 1, 10)
        if probs_synth.shape[0] > 0:
            synth_gt_hist, _ = np.histogram(probs_synth[:, 0], bins=bins, density=True)
            synth_bg_hist, _ = np.histogram(probs_synth[:, 1], bins=bins, density=True)
            ax.bar(bins[:-1], synth_gt_hist, width=0.02, color='b', label='Synthetic-GT')
            ax.bar(bins[:-1] + 0.02, synth_bg_hist, width=0.02, color='g', label='Synthetic-BG')
        if probs_real.shape[0] > 0:
            real_gt_hist, _ = np.histogram(probs_real[:, 0], bins=bins, density=True)
            real_bg_hist, _ = np.histogram(probs_real[:, 1], bins=bins, density=True)
            ax.bar(bins[:-1] + 0.04, real_gt_hist, width=0.02, color='r', label='Real-GT')
            ax.bar(bins[:-1] + 0.06, real_bg_hist, width=0.02, color='c', label='Real-BG')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.legend()
        wandb.log({"Probability Distribution for {}".format(c_name): wandb.Image(plt)})
        plt.close()

        # accuracy_rates = [cor / total if total > 0 else 0 for cor, total in zip(stat['correct'], stat['total'])]
        # custom_chart = wandb.plot_table(
        #     "custom_chart",
        #     data=[[category, total, accuracy] for category, total, accuracy in
        #           zip(class_names, stat['total'], accuracy_rates)],
        #     columns=["Category", "Total Count", "Accuracy Rate"]
        # )
        # wandb.log({"Total Count and Accuracy Rate for {} Objects".format(type_name): custom_chart})

        # for cls_name, probabilities in zip(class_names, stat["prob"]):
        #     # table = wandb.Table(data=probabilities.tolist(), columns=["scores"])
        #     wandb.log({f"{type_name}_{cls_name}_probability_distribution": wandb.Histogram(probabilities.tolist())})


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Load pre-trained object detector
    cfg = setup(args)
    wandb.run.name = cfg.OUTPUT_DIR

    # Load Object Detector
    detector = build_model(cfg)
    DetectionCheckpointer(detector, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    detector.eval()
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0])
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    #
    # features = [torch.Tensor([]).to(detector.device) for _ in class_names]
    # # Extract GT features
    # with torch.no_grad():
    #     for idx, inputs in enumerate(data_loader):
    #         if idx % 10 == 0:
    #             print(idx)
    #         if idx > 500:
    #             break
    #         box_features, preds, gt_classes = detector.extract_features(inputs)
    #         for i, cls in enumerate(class_names):
    #             features[i] = torch.cat([features[i], box_features[gt_classes == i].mean(dim=[2,3])], dim=0)
    #             # predictions[i] = torch.cat([predictions[i], preds[0][gt_classes == i].argmax(dim=1)])
    #     #     print('')

    # Load pre-trained GLIGEN diffusion
    pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Extract Noised Image Features
    # pipe.scheduler.set_timesteps(50, device="cuda")
    # timesteps = pipe.scheduler.timesteps
    #
    # for idx, inputs in enumerate(data_loader):
    #     img1, img2 = inputs[0]['image'][:, :, :600], inputs[0]['image'][:, :, 600:]
    #     gligen_inpaint_image = to_pil_image(img1)
    #     gligen_inpaint_image = pipe.target_size_center_crop(gligen_inpaint_image, pipe.vae.sample_size)
    #     gligen_inpaint_image = pipe.image_processor.preprocess(gligen_inpaint_image)
    #     gligen_inpaint_image = gligen_inpaint_image.to(dtype=pipe.vae.dtype, device=pipe.vae.device)
    #     gligen_inpaint_latent = pipe.vae.encode(gligen_inpaint_image).latent_dist.sample()
    #     gligen_inpaint_latent = pipe.vae.config.scaling_factor * gligen_inpaint_latent
    #     for t in timesteps:
    #         gligen_inpaint_latent_with_noise = (
    #             pipe.scheduler.add_noise(
    #                 gligen_inpaint_latent, torch.randn_like(gligen_inpaint_latent), torch.tensor([t])
    #             )
    #                 .expand(latents.shape[0], -1, -1, -1)
    #                 .clone())

    if args.feats_path != "":
        feats_bank = torch.load(args.feats_path)
    else:
        feats_bank = pipe.extract_noised_features(data_loader, detector, wandb=wandb)
    generate_image_with_detector_guidance(cfg, pipe, detector, data_loader, class_names, feats_bank=feats_bank, loss_type=args.loss_type, guidance_step=args.guidance_step, guidance_weight=args.guidance_weight)
    wandb.finish()
    return

# # Placeholder for bounding box and class information
# bounding_boxes = [...]  # Format: [[x_min, y_min, x_max, y_max], ...]
# target_classes = [...]  # The target classes for each bounding box
#
# # Initialize the diffusion pipeline with your model
# pipeline = DiffusionPipeline(model=gligen_model)
#
#
# # Function to compute object detector loss
# def compute_detector_loss(image, target_classes, bounding_boxes):
#     # Perform object detection
#     predictions = object_detector(image, bounding_boxes)
#
#     # Compute loss based on how well the predictions match target classes
#     loss = ...  # Define your loss function based on predictions and target_classes
#     return loss
#
#
# # Custom denoise function with gradient ascent
# def custom_denoise_step(model, latents, timesteps, **kwargs):
#     latents.requires_grad_(True)
#     optimizer = torch.optim.Adam([latents], lr=0.01)  # Use an appropriate learning rate
#
#     for step in range(grad_ascent_steps):
#         optimizer.zero_grad()
#
#         # Generate image from current latents
#         with torch.no_grad():
#             pred_image = model.decode(latents, timesteps)
#
#         # Compute detector loss
#         loss = compute_detector_loss(pred_image, target_classes, bounding_boxes)
#         (-loss).backward()  # Maximize the classification accuracy (minimize negative loss)
#         optimizer.step()
#
#     with torch.no_grad():
#         # Update latents based on optimized gradient ascent
#         updated_latents = model.denoise_step(latents, timesteps, **kwargs)
#
#     return updated_latents
#
#
# # Assuming `generate_with_custom_denoise` is a function you modify or create within the GLIGEN pipeline
# # to use `custom_denoise_step` during the diffusion process
# generated_images = pipeline.generate_with_custom_denoise(bounding_boxes, custom_denoise_fn=custom_denoise_step)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
