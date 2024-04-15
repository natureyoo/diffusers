import torch
from diffusers import StableDiffusionGLIGENTextImagePipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw
import random, os

# Insert objects described by image at the region defined by bounding boxes
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
    "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# input_image = load_image(
#     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
# )
cs_list = ["leftImg8bit/test/munich/munich_000208_000019_leftImg8bit.png", "leftImg8bit/test/munich/munich_000067_000019_leftImg8bit.png", "leftImg8bit/test/bonn/bonn_000021_000019_leftImg8bit.png", "leftImg8bit/test/leverkusen/leverkusen_000056_000019_leftImg8bit.png", "leftImg8bit/test/bielefeld/bielefeld_000000_047918_leftImg8bit.png",
           "leftImg8bit/test/bielefeld/bielefeld_000000_028550_leftImg8bit.png", "leftImg8bit/test/mainz/mainz_000001_048725_leftImg8bit.png", "leftImg8bit/test/mainz/mainz_000001_028566_leftImg8bit.png", "leftImg8bit/test/berlin/berlin_000492_000019_leftImg8bit.png", "leftImg8bit/test/berlin/berlin_000540_000019_leftImg8bit.png", ]
foggy_list = ["leftImg8bit_foggy/val/munster/munster_000021_000019_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/val/munster/munster_000103_000019_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/val/munster/munster_000049_000019_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/val/munster/munster_000083_000019_leftImg8bit_foggy_beta_0.02.png",
              "leftImg8bit_foggy/val/frankfurt/frankfurt_000001_048654_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/val/frankfurt/frankfurt_000000_014480_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/test/berlin/berlin_000044_000019_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/test/berlin/berlin_000507_000019_leftImg8bit_foggy_beta_0.02.png",
              "leftImg8bit_foggy/test/berlin/berlin_000489_000019_leftImg8bit_foggy_beta_0.02.png", "leftImg8bit_foggy/test/bielefeld/bielefeld_000000_000856_leftImg8bit_foggy_beta_0.02.png"]
voc_list = ["2007_000027.jpg",
"2007_000032.jpg",
"2007_000033.jpg",
"2007_000039.jpg",
"2007_000042.jpg",
"2007_000061.jpg",
"2007_000063.jpg",
"2007_000068.jpg",]
# for i in range(10):
# for i, _img in enumerate(cs_list):
# obj_file_name = ['fire-hydrant.jpeg', 'traffic-sign.jpeg', 'traffic-light.jpeg']
# gligen_images = [Image.open(os.path.join("./asset", im)) for im in obj_file_name]
for i, img in enumerate(foggy_list):
    # input_image = Image.open("/data/Dataset/KITTI/data_object_label_2/training/image_2/00000{}.png".format(i))
    input_image = Image.open(os.path.join("/data/Dataset/cityscapes", img))
    # input_image = Image.open(os.path.join("/data/Dataset/VOCdevkit/VOC2012/JPEGImages", img))
    prompt = "a photo of street"
    # boxes = [[0.2676, 0.4088, 0.3773, 0.7183]]
    boxes = []
    for _ in range(3):
        x, y, w, h = random.random(), random.random(), random.random() * 0.5 + 0.1, random.random() * 0.5 + 0.1
        box = [x, y, min(x + w, 0.99), min(y + h, 0.99)]
        boxes.append(box)
    phrases = ["fire hydrant", "traffic light", "traffic sign"]
    gligen_images = None
    # phrases = ["bottle", "sofa", "person"]
    # phrases = None
    # gligen_image = load_image(
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
    # )
    images = pipe(
        prompt=prompt,
        gligen_phrases=phrases,
        gligen_inpaint_image=input_image,
        gligen_boxes=boxes,
        gligen_images=gligen_images,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=50,
    ).images

    w_, h_ = input_image.size
    paste_pos = ((w_ - min(h_, w_)) // 2, (h_ - min(h_, w_)) // 2)
    input_image.paste(images[0].resize((min(h_, w_), min(h_, w_))), paste_pos)
    # input_image.paste(images[0].resize((1024, 1024)), (512, 0))
    input_image.save("./gligen-foggy-cs-{}-original_size.jpg".format(i))

    draw = ImageDraw.Draw(images[0])
    h, w = images[0].size
    for box, phr in zip(boxes, phrases):
    # for box, _phr in zip(boxes, obj_file_name):
    #     phr = _phr.replace('.jpeg', '')
        draw.rectangle((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)), outline=(0, 255, 0), width=3)
        draw.text((int(box[0] * w) + 3, int(box[1] * h) + 3), phr, (0, 255, 0))
    input_image.paste(images[0].resize((min(h_, w_), min(h_, w_))), paste_pos)
    # input_image.paste(images[0].resize((1024, 1024)), (512, 0))
    input_image.save("./gligen-foggy-cs-{}-original_size-GT.jpg".format(i))

# Generate an image described by the prompt and
# insert objects described by text and image at the region defined by bounding boxes
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
    "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a flower sitting on the beach"
boxes = [[0.0, 0.09, 0.53, 0.76]]
phrases = ["flower"]
gligen_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/pexels-pixabay-60597.jpg"
)

images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_images=[gligen_image],
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
).images

images[0].save("./gligen-generation-text-image-box.jpg")

# Generate an image described by the prompt and
# transfer style described by image at the region defined by bounding boxes
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
    "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a dragon flying on the sky"
boxes = [[0.4, 0.2, 1.0, 0.8], [0.0, 1.0, 0.0, 1.0]]  # Set `[0.0, 1.0, 0.0, 1.0]` for the style

gligen_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)

gligen_placeholder = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)

images = pipe(
    prompt=prompt,
    gligen_phrases=[
        "dragon",
        "placeholder",
    ],  # Can use any text instead of `placeholder` token, because we will use mask here
    gligen_images=[
        gligen_placeholder,
        gligen_image,
    ],  # Can use any image in gligen_placeholder, because we will use mask here
    input_phrases_mask=[1, 0],  # Set 0 for the placeholder token
    input_images_mask=[0, 1],  # Set 0 for the placeholder image
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
).images

images[0].save("./gligen-generation-text-image-box-style-transfer.jpg")