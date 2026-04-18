"""Generation pipeline for multimodal-conditioned medical image synthesis.

Adapted from the author's internal script for SDXL + LoRA + IP-Adapter based
medical image generation. The released version exposes a clean CLI and removes
hard-coded experiment paths.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from diffusers import AutoPipelineForText2Image


class GenerationConfig:
    def __init__(
        self,
        pretrained_lora_path,
        pretrained_model_name_or_path,
        ip_adapter_path,
        save_dir,
        image_encoder_path,
        data_path,
    ):
        self.pretrained_lora_path = pretrained_lora_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.ip_adapter_path = ip_adapter_path
        self.save_dir = save_dir
        self.image_encoder_path = image_encoder_path
        self.data_path = data_path


def build_pipeline(args, device):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    if args.ip_adapter_path:
        pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict=args.ip_adapter_path,
            weight_name="ip_adapter.safetensors",
            subfolder="",
            image_encoder_folder=args.image_encoder_path,
        )
        pipeline.set_ip_adapter_scale(args.ip_adapter_scale)

    if args.pretrained_lora_path:
        pipeline.load_lora_weights(args.pretrained_lora_path, adapter_name="medical")
        pipeline.set_adapters(["medical"], adapter_weights=[args.lora_weight])

    return pipeline


def resolve_condition_image(mask_root, item):
    image_name = item["image"]
    return Path(mask_root) / image_name


def generate(args):
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = build_pipeline(args, device)
    resize_transform = transforms.Resize(
        args.image_size,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    metadata_path = Path(args.save_dir) / "_data.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for item in data:
            cond_path = resolve_condition_image(args.condition_image_root, item)
            image = Image.open(cond_path).convert("RGB")
            image = resize_transform(image)
            prompt = item["description"]

            result = pipeline(
                prompt=prompt,
                ip_adapter_image=image,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                do_classifier_free_guidance=True,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                num_images_per_prompt=1,
            ).images[0]

            filename = item["image"]
            result.save(os.path.join(args.save_dir, filename))
            metadata_file.write(
                json.dumps({"image": filename, "description": prompt}, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic medical images.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_lora_path", type=str, default=None)
    parser.add_argument("--ip_adapter_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True, help="Input JSONL with image and description fields.")
    parser.add_argument("--condition_image_root", type=str, required=True, help="Directory containing conditioning images or masks.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--num_inference_steps", type=int, default=60)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.8)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Distorted, blurred, incomplete, wrong proportions, low resolution, bad anatomy, worst quality, low quality",
    )
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    cli_args = parser.parse_args()

    print(f"seed: {cli_args.seed}")
    generate(cli_args)
