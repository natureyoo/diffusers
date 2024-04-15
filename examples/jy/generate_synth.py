import torch
from diffusers import StableDiffusionGLIGENTextImagePipeline, StableDiffusionGLIGENPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw
import random, os, sys

save_dir = "/data/synthetic_foggy/20240327_inpainting"

# Insert objects described by image at the region defined by bounding boxes
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
# pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
    # "masterful/gligen-1-4-inpainting-text-box", variant="fp16", torch_dtype=torch.float16
    # "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 500 reference images
file_list = []
with open("/data/jayeon/irg-sfda/dataset/cityscape/VOC2007/ImageSets/Main/train_t.txt", "r") as f:
    for l in f.readlines():
        file_list.append(l.strip())

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "Annotations"), exist_ok=True)

# random.shuffle(file_list)
novel_classes = ['a person who is walking', 'a person who is riding a bicycle', 'a car', 'a truck', 'a bus', 'a train', 'a motorcycle', 'a bicycle']
prompt = "a photo of city street"
s_idx = int(sys.argv[1])
file_list = file_list[len(file_list)*s_idx//4:len(file_list)*(s_idx + 1)//4]
for f in file_list:
    input_image = Image.open(os.path.join("/data/Dataset/cityscapes_vocstyle/VOC2007/JPEGImages", "{}.jpg".format(f)))

    img1 = input_image.crop((0, 0, 1024, 1024))
    img2 = input_image.crop((1024, 0, 2048, 1024))
    # insert three novel class objects
    new_imgs = []
    new_boxes = []
    new_classes = []
    for idx, img in enumerate([img1, img2]):
        boxes = []
        for _ in range(3):
            x, y, w, h = random.random(), random.random(), random.random() * 0.2 + 0.1, random.random() * 0.2 + 0.1
            box = [max(min(x, 0.9), 0.01), max(min(y, 0.9), 0.01), min(x + w, 0.99), min(y + h, 0.99)]
            boxes.append(box)
        phrases = random.choices(novel_classes, k=3)
        prompt = " and ".join(phrases) + " in the street"
        gligen_images = None
        # phrases = ["bottle", "sofa", "person"]
        # phrases = None
        # gligen_image = load_image(
        #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
        # )
        images = pipe(
            prompt=prompt,
            gligen_phrases=phrases,
            gligen_inpaint_image=img,
            # gligen_inpaint_image=None,
            gligen_boxes=boxes,
            # gligen_images=gligen_images,
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=50,
        ).images
        new_imgs.append(images[0].resize((1024, 1024)))
        new_boxes += boxes if idx == 0 else [[b[0] + 1, b[1], b[2] + 1, b[3]] for b in boxes]
        new_classes += phrases

    new_img = Image.new('RGB', (2048, 1024))
    new_img.paste(new_imgs[0], (0, 0))
    new_img.paste(new_imgs[1], (1024, 0))
    new_img.save(os.path.join(save_dir, "JPEGImages/{}.jpg".format(f)))
    # images[0].show()
    # save images
    # w_, h_ = input_image.size
    # paste_pos = ((w_ - min(h_, w_)) // 2, (h_ - min(h_, w_)) // 2)
    # input_image.paste(images[0].resize((min(h_, w_), min(h_, w_))), paste_pos)
    # input_image.paste(images[0].resize((1024, 1024)), (512, 0))
    # input_image.save(os.path.join(save_dir, "JPEGImages2/{}.jpg".format(f)))
    with open(os.path.join(save_dir, "Annotations/{}.txt".format(f)), "w") as ann:
        for cls, box in zip(new_classes, new_boxes):
            x1 = int(box[0] * 1024)
            y1 = int(box[1] * 1024)
            x2 = int(box[2] * 1024)
            y2 = int(box[3] * 1024)
            ann.write("{},{},{},{},{}\n".format(cls, x1, y1, x2, y2))
    # save annotations
    # with open(os.path.join(save_dir, "Annotations2/{}.txt".format(f)), "w") as ann:
    #     for cls, box in zip(phrases, boxes):
    #         x1 = int(box[0] * 512 + paste_pos[0])
    #         y1 = int(box[1] * 512 + paste_pos[1])
    #         x2 = int(box[2] * 512 + paste_pos[0])
    #         y2 = int(box[3] * 512 + paste_pos[1])
    #         ann.write("{},{},{},{},{}\n".format(cls, x1, y1, x2, y2))
    # draw = ImageDraw.Draw(images[0])
    # h, w = images[0].size
    # for box, phr in zip(boxes, phrases):
    # # for box, _phr in zip(boxes, obj_file_name):
    # #     phr = _phr.replace('.jpeg', '')
    #     draw.rectangle((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)), outline=(0, 255, 0), width=3)
    #     draw.text((int(box[0] * w) + 3, int(box[1] * h) + 3), phr, (0, 255, 0))
    # input_image.paste(images[0].resize((min(h_, w_), min(h_, w_))), paste_pos)
    # # input_image.paste(images[0].resize((1024, 1024)), (512, 0))
    # input_image.save("./gligen-foggy-cs-{}-original_size-GT.jpg".format(i))
