#!/usr/bin/env python3
"""
Circle-Rotate Inference Script

Dual-stage LoRA inference for rigid-body video generation.
Based on Wan2.2 Fun Inpaint model with temporal handover mechanism.

Requirements:
    - ComfyUI installed (https://github.com/comfyanonymous/ComfyUI)
    - Required models downloaded (see README.md)

Usage:
    python inference.py \
        --first_frame examples/first.png \
        --last_frame examples/last.png \
        --prompt "A car, camera smoothly orbits left" \
        --output outputs/result.mp4
"""

import argparse
import os
import sys

# Add ComfyUI to path
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "../ComfyUI")
if os.path.exists(COMFYUI_PATH):
    sys.path.insert(0, COMFYUI_PATH)

import torch
import numpy as np
from PIL import Image


def load_image(image_path: str) -> torch.Tensor:
    """Load image and convert to tensor."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0)


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
    """Save frames as video."""
    import cv2

    frames_np = (frames.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    h, w = frames_np.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_np:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved to {output_path}")


class CircleRotateInference:
    """
    Dual-stage LoRA inference for rigid-body video generation.

    Configuration (matching paper):
        - CFG: 1.6
        - Steps: 4 (handover at step 2)
        - Resolution: 720x1280, 81 frames
        - Sampler: euler, Scheduler: simple, Shift: 8
    """

    def __init__(
        self,
        lora_high: str = "circle_rotate_h.safetensors",
        lora_low: str = "circle_rotate_l.safetensors",
        device: str = "cuda",
    ):
        self.device = device
        self.lora_high = lora_high
        self.lora_low = lora_low
        self._loaded = False

    def _load_models(self):
        """Load all required models using ComfyUI nodes."""
        if self._loaded:
            return

        from nodes import NODE_CLASS_MAPPINGS

        # Get node classes
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        self.UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.ModelSamplingSD3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        self.CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.WanFunInpaintToVideo = NODE_CLASS_MAPPINGS["WanFunInpaintToVideo"]()
        self.KSamplerAdvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        print("Loading VAE...")
        self.vae = self.VAELoader.load_vae("wan_2.1_vae.safetensors")[0]

        print("Loading CLIP...")
        self.clip = self.CLIPLoader.load_clip(
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"
        )[0]

        print("Loading High-Noise UNET...")
        model_high = self.UNETLoader.load_unet(
            "wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors", "default"
        )[0]

        print("Loading Low-Noise UNET...")
        model_low = self.UNETLoader.load_unet(
            "wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors", "default"
        )[0]

        self._loaded = True
        self._model_high_base = model_high
        self._model_low_base = model_low
        print("Base models loaded.")

    def _apply_loras(self, lora_strength: float = 1.0):
        """Apply LoRA adapters to models."""
        print(f"Loading LoRAs (strength={lora_strength})...")

        # High-noise branch: 4-step LoRA -> Circle-Rotate LoRA
        model_h = self.LoraLoaderModelOnly.load_lora_model_only(
            self._model_high_base,
            "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            1.0
        )[0]
        model_h = self.LoraLoaderModelOnly.load_lora_model_only(
            model_h, self.lora_high, lora_strength
        )[0]
        self.model_high = self.ModelSamplingSD3.patch(model_h, 8.0)[0]

        # Low-noise branch: 4-step LoRA -> Circle-Rotate LoRA
        model_l = self.LoraLoaderModelOnly.load_lora_model_only(
            self._model_low_base,
            "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            1.0
        )[0]
        model_l = self.LoraLoaderModelOnly.load_lora_model_only(
            model_l, self.lora_low, lora_strength
        )[0]
        self.model_low = self.ModelSamplingSD3.patch(model_l, 8.0)[0]

        print("LoRAs applied.")

    def generate(
        self,
        first_frame: str,
        last_frame: str,
        prompt: str = "A static object, camera smoothly orbits around it",
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,
        steps: int = 4,
        cfg: float = 1.6,
        seed: int = None,
        lora_strength: float = 1.0,
    ) -> torch.Tensor:
        """Generate video with dual-stage inference."""
        self._load_models()
        self._apply_loras(lora_strength)

        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()

        # Load images
        print("Loading images...")
        start_img = load_image(first_frame)
        end_img = load_image(last_frame)

        # Encode prompts
        print("Encoding prompts...")
        pos_cond = self.CLIPTextEncode.encode(self.clip, prompt)[0]
        neg_cond = self.CLIPTextEncode.encode(self.clip, negative_prompt)[0]

        # Prepare video conditioning
        print(f"Preparing {num_frames} frames at {width}x{height}...")
        pos, neg, latent = self.WanFunInpaintToVideo.encode(
            pos_cond, neg_cond, self.vae, None,
            start_img, end_img, width, height, num_frames, 1
        )

        # Stage 1: High-noise sampling (steps 0-2)
        print("Stage 1: High-noise sampling...")
        latent = self.KSamplerAdvanced.sample(
            self.model_high, "enable", seed, steps, cfg,
            "euler", "simple", pos, neg, latent,
            0, 2, "enable"
        )[0]

        # Stage 2: Low-noise sampling (steps 2-4)
        print("Stage 2: Low-noise sampling...")
        latent = self.KSamplerAdvanced.sample(
            self.model_low, "disable", 0, steps, cfg,
            "euler", "simple", pos, neg, latent,
            2, steps, "disable"
        )[0]

        # Decode
        print("Decoding video...")
        frames = self.VAEDecode.decode(self.vae, latent)[0]

        print("Generation complete.")
        return frames


def main():
    parser = argparse.ArgumentParser(
        description="Circle-Rotate: Rigid-body video generation"
    )

    parser.add_argument("-f", "--first_frame", required=True, help="First frame path")
    parser.add_argument("-l", "--last_frame", required=True, help="Last frame path")
    parser.add_argument("-p", "--prompt", default="A static object, camera smoothly orbits around it")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("--lora_high", default="circle_rotate_h.safetensors")
    parser.add_argument("--lora_low", default="circle_rotate_l.safetensors")
    parser.add_argument("--lora_strength", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.first_frame):
        raise FileNotFoundError(f"First frame not found: {args.first_frame}")
    if not os.path.exists(args.last_frame):
        raise FileNotFoundError(f"Last frame not found: {args.last_frame}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Generate
    model = CircleRotateInference(
        lora_high=args.lora_high,
        lora_low=args.lora_low,
    )

    frames = model.generate(
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        lora_strength=args.lora_strength,
    )

    save_video(frames, args.output, args.fps)


if __name__ == "__main__":
    main()
