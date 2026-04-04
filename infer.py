import argparse

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_QUESTIONS = {
    "depth": "Looking at this image, which colored point appears to be closer to the camera - red or green?",
    "normal": "Which point has a surface more facing towards the camera, red or green?",
    "albedo": "Do the red and green points have the same base color?",
    "irradiance": "Which point is more illuminated?",
}


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=512, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=512, max_num=12):
    """accept a file path (str) or a PIL Image."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def draw_points(image, red_xy, green_xy, circle_radius=5.0, center_radius=1.0, line_width=1.0):
    """draw hollow red and green circle markers on a PIL image."""
    rgb_np = np.array(image).astype(np.float32)
    rgb_tensor = torch.from_numpy(rgb_np / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

    h, w = rgb_tensor.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing='ij')

    for (px, py), color in [(red_xy, (255, 0, 0)), (green_xy, (0, 255, 0))]:
        dist = torch.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        mask = torch.clamp(
            (torch.abs(dist - circle_radius) <= line_width).float() +
            (dist <= center_radius).float(), 0, 1)
        color_t = torch.tensor(color, dtype=torch.float32) / 127.5 - 1.0
        for c in range(3):
            rgb_tensor[0, c] = rgb_tensor[0, c] * (1 - mask) + color_t[c] * mask

    rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)
    out = ((rgb_tensor[0].permute(1, 2, 0) + 1.0) * 127.5).numpy().astype(np.uint8)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser(description='Run the intrinsic judge on a single image.')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model_path', default='adirik/InternVL2_5-4B-Intrinsic-Judge',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--question', type=str, default=None,
                        help='Question to ask (omit together with --all to ask all four)')
    parser.add_argument('--all', action='store_true',
                        help='Ask all four default modality questions')
    parser.add_argument('--red', type=int, nargs=2, metavar=('X', 'Y'), default=None,
                        help='Red point pixel coordinates (draws marker on image)')
    parser.add_argument('--green', type=int, nargs=2, metavar=('X', 'Y'), default=None,
                        help='Green point pixel coordinates (draws marker on image)')
    parser.add_argument('--save_annotated', type=str, default=None,
                        help='If set, save the annotated image to this path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    if args.question is None and not args.all:
        parser.error('Provide --question or --all')

    pil_image = Image.open(args.image).convert('RGB')

    if args.red is not None and args.green is not None:
        pil_image = draw_points(pil_image, tuple(args.red), tuple(args.green))
        print(f"Drew red marker at {tuple(args.red)}, green marker at {tuple(args.green)}")

    if args.save_annotated:
        pil_image.save(args.save_annotated)
        print(f"Saved annotated image to {args.save_annotated}")

    pixel_values = load_image(pil_image).to(torch.bfloat16)
    if args.device == 'cuda':
        pixel_values = pixel_values.cuda()

    print(f"Loading model: {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

    if args.device == 'cuda':
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=128, do_sample=False)
    questions = (
        list(DEFAULT_QUESTIONS.values()) if args.all
        else [args.question]
    )

    for q in questions:
        prompt = f"<image>\n{q}"
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        print(f"\nQ: {q}")
        print(f"A: {response}")


if __name__ == '__main__':
    main()
