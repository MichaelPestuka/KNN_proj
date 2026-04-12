from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.amp import autocast
from torchvision import transforms

from config import (
    BUFFER_SIZE,
    CFG_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    HEIGHT,
    NUM_ACTIONS,
    WIDTH,
)
from model import load_model

torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)

_START_IMAGE = Path(__file__).resolve().parent / "sample_images" / "start.jpg"


def _autocast_dtype(device: torch.device) -> torch.dtype:
    """Use FP16 autocast on CUDA; keep FP32 elsewhere (MPS/CPU)."""
    return torch.float16 if device.type == "cuda" else torch.float32


def encode_conditioning_frames(
    vae: AutoencoderKL, images: torch.Tensor, vae_scale_factor: int, dtype: torch.dtype
) -> torch.Tensor:
    batch_size, _, channels, height, width = images.shape
    context_frames = images[:, :BUFFER_SIZE].reshape(-1, channels, height, width)
    conditioning_frame_latents = vae.encode(
        context_frames.to(device=vae.device, dtype=dtype)
    ).latent_dist.sample()
    conditioning_frame_latents = conditioning_frame_latents * vae.config.scaling_factor

    # Reshape context latents
    conditioning_frame_latents = conditioning_frame_latents.reshape(
        batch_size,
        BUFFER_SIZE,
        vae.config.latent_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    return conditioning_frame_latents


def get_initial_noisy_latent(
    noise_scheduler: DDPMScheduler,
    batch_size: int,
    height: int,
    width: int,
    num_channels_latents: int,
    vae_scale_factor: int,
    device: torch.device,
    dtype=torch.float32,
):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * noise_scheduler.init_noise_sigma
    return latents


def next_latent(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    action_embedding: torch.nn.Embedding,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = CFG_GUIDANCE_SCALE,
    skip_action_conditioning: bool = False,
    *,
    precomputed_timesteps: torch.Tensor | None = None,
    class_labels: torch.Tensor | None = None,
    verify_context: bool = False,
):
    batch_size = context_latents.shape[0]
    latent_height = context_latents.shape[-2]
    latent_width = context_latents.shape[-1]
    num_channels_latents = context_latents.shape[2]

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    device_type = (
        device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    )
    amp_dtype = _autocast_dtype(device)
    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype):
        # Generate initial noise for the target frame
        latents = get_initial_noisy_latent(
            noise_scheduler,
            batch_size,
            HEIGHT,
            WIDTH,
            num_channels_latents,
            vae_scale_factor,
            device,
            dtype=unet.dtype,
        )

        # Prepare timesteps (caller may cache identical schedule across frames)
        if precomputed_timesteps is not None:
            timesteps = precomputed_timesteps
        else:
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = noise_scheduler.timesteps

        if not skip_action_conditioning:
            if do_classifier_free_guidance:
                encoder_hidden_states = action_embedding(actions.to(device)).repeat(
                    2, 1, 1
                )
            else:
                encoder_hidden_states = action_embedding(actions.to(device))

        latents = torch.cat([context_latents, latents.unsqueeze(1)], dim=1)

        # Fold the conditioning frames into the channel dimension
        latents = latents.view(batch_size, -1, latent_height, latent_width)

        # Denoising loop
        for _, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                # In case of classifier free guidance, the unconditional case is without conditioning frames
                uncond_latents = latents.clone()
                uncond_latents[:, :BUFFER_SIZE] = torch.zeros_like(
                    uncond_latents[:, :BUFFER_SIZE]
                )
                # BEWARE: order is important, the unconditional case should come first
                latent_model_input = torch.cat([uncond_latents, latents])
            else:
                latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            cl_batch = latent_model_input.shape[0]
            if class_labels is not None and class_labels.shape[0] == cl_batch:
                cl = class_labels
            else:
                cl = torch.zeros(cl_batch, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=cl,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Perform denoising step on the last frame only
            reshaped_frames = latents.reshape(
                batch_size,
                BUFFER_SIZE + 1,
                num_channels_latents,
                latent_height,
                latent_width,
            )
            last_frame = reshaped_frames[:, -1]
            denoised_last_frame = noise_scheduler.step(
                noise_pred, t, last_frame, return_dict=False
            )[0]

            reshaped_frames[:, -1] = denoised_last_frame
            latents = reshaped_frames.reshape(
                batch_size, -1, latent_height, latent_width
            )

            # The conditioning frames should not be modified by the denoising process
            if verify_context:
                assert torch.all(context_latents == reshaped_frames[:, :BUFFER_SIZE])

        # Return the final latents of the target frame only
        reshaped_frames = latents.reshape(
            batch_size,
            BUFFER_SIZE + 1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        return reshaped_frames[:, -1]


def decode_and_postprocess(
    vae: AutoencoderKL, image_processor: VaeImageProcessor, latents: torch.Tensor
) -> Image:
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    image = image_processor.postprocess(
        image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
    )[0]
    return image


def run_inference_img_conditioning_with_params(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    tokenizer,
    text_encoder,
    batch,
    device,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance=True,
    guidance_scale=CFG_GUIDANCE_SCALE,
    skip_action_conditioning=False,
    image_processor: VaeImageProcessor | None = None,
) -> Image:
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if image_processor is None:
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    device_type = (
        device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    )
    amp_dtype = _autocast_dtype(device)
    encode_dtype = vae.dtype
    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype):
        actions = batch["input_ids"]

        conditioning_frames_latents = encode_conditioning_frames(
            vae,
            images=batch["pixel_values"],
            vae_scale_factor=vae_scale_factor,
            dtype=encode_dtype,
        )
        new_frame = next_latent(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=conditioning_frames_latents,
            device=device,
            actions=actions,
            skip_action_conditioning=skip_action_conditioning,
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
        )

        # only take the last frame
        image = decode_and_postprocess(
            vae=vae, image_processor=image_processor, latents=new_frame
        )
    return image


def _default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def build_batch_from_start_image() -> dict:
    """All BUFFER_SIZE+1 frames are the same image (start.jpg)."""
    img = Image.open(_START_IMAGE).convert("RGB")
    transform = _default_image_transform()
    frame = transform(img)
    frames = torch.stack([frame] * (BUFFER_SIZE + 1)).unsqueeze(0).float()
    actions = torch.zeros(1, BUFFER_SIZE + 1, dtype=torch.long)
    return {"pixel_values": frames, "input_ids": actions}


def build_batch_from_buffers(
    frame_buffer: list[Image.Image],
    action_buffer: list[int],
    new_action: int,
) -> dict:
    """
    Build a batch for one autoregressive step.

    ``frame_buffer`` and ``action_buffer`` each have length BUFFER_SIZE (9).
    The model consumes the first BUFFER_SIZE frames as conditioning; the
    (BUFFER_SIZE+1)-th tensor slot is kept for shape compatibility (duplicate
    of the last context frame), matching ``build_batch_from_start_image``.
    """
    if len(frame_buffer) != BUFFER_SIZE or len(action_buffer) != BUFFER_SIZE:
        raise ValueError(
            f"Expected {BUFFER_SIZE} frames and {BUFFER_SIZE} past actions, "
            f"got {len(frame_buffer)} and {len(action_buffer)}"
        )
    transform = _default_image_transform()
    tensors: list[torch.Tensor] = [transform(img) for img in frame_buffer]
    tensors.append(tensors[-1].clone())
    frames = torch.stack(tensors).unsqueeze(0).float()
    actions = torch.tensor(action_buffer + [new_action], dtype=torch.long).unsqueeze(0)
    return {"pixel_values": frames, "input_ids": actions}


def _encode_initial_latent_buffer(
    vae: AutoencoderKL,
    img_transform: transforms.Compose,
    start_image: Image.Image,
    device: torch.device,
    vae_scale_factor: int,
) -> torch.Tensor:
    """Encode BUFFER_SIZE copies of the start frame to latent space [BUFFER_SIZE, C, H, W]."""
    frame = img_transform(start_image.convert("RGB"))
    # [1, BUFFER_SIZE, 3, H, W] — same as training batch with repeated start
    pixel_batch = frame.unsqueeze(0).expand(BUFFER_SIZE, -1, -1, -1).unsqueeze(0)
    encode_dtype = vae.dtype
    device_type = (
        device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    )
    amp_dtype = _autocast_dtype(device)
    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype):
        conditioning = encode_conditioning_frames(
            vae,
            images=pixel_batch.to(device=device, dtype=encode_dtype),
            vae_scale_factor=vae_scale_factor,
            dtype=encode_dtype,
        )
    return conditioning.squeeze(0)


class InferenceEngine:
    """Rolling-window GameNGen inference: 9 past frames/actions + current action -> next frame."""

    def __init__(
        self,
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        device: torch.device,
        *,
        start_image_path: Path | None = None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = CFG_GUIDANCE_SCALE,
        skip_action_conditioning: bool = False,
    ) -> None:
        self._unet = unet
        self._vae = vae
        self._noise_scheduler = noise_scheduler
        self._action_embedding = action_embedding
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self._device = device
        self._num_inference_steps = num_inference_steps
        self._do_cfg = do_classifier_free_guidance
        self._guidance_scale = guidance_scale
        self._skip_action_conditioning = skip_action_conditioning

        self._vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)
        self._img_transform = _default_image_transform()

        # Scheduler timesteps are identical every frame — compute once
        self._noise_scheduler.set_timesteps(num_inference_steps, device=device)
        self._timesteps = self._noise_scheduler.timesteps

        # class_labels batch size matches UNet input (1 without CFG, 2 with CFG)
        cl_n = 2 if do_classifier_free_guidance else 1
        self._class_labels = torch.zeros(cl_n, dtype=torch.long, device=device)

        path = start_image_path or _START_IMAGE
        start = Image.open(path).convert("RGB")
        self._latent_buffer = _encode_initial_latent_buffer(
            vae,
            self._img_transform,
            start,
            device,
            self._vae_scale_factor,
        )
        self._action_buffer: list[int] = [0] * BUFFER_SIZE

    def reset(self, start_image_path: Path | None = None) -> None:
        path = start_image_path or _START_IMAGE
        start = Image.open(path).convert("RGB")
        self._latent_buffer = _encode_initial_latent_buffer(
            self._vae,
            self._img_transform,
            start,
            self._device,
            self._vae_scale_factor,
        )
        self._action_buffer = [0] * BUFFER_SIZE

    def step(self, action_index: int) -> Image.Image:
        if action_index < 0 or action_index >= NUM_ACTIONS:
            raise ValueError(
                f"action_index must be in [0, {NUM_ACTIONS}), got {action_index}"
            )

        actions = (
            torch.tensor(
                self._action_buffer + [action_index], dtype=torch.long, device=self._device
            )
            .unsqueeze(0)
        )

        device_type = (
            self._device.type
            if isinstance(self._device, torch.device)
            else str(self._device).split(":")[0]
        )
        amp_dtype = _autocast_dtype(self._device)

        with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype):
            new_latent = next_latent(
                unet=self._unet,
                vae=self._vae,
                noise_scheduler=self._noise_scheduler,
                action_embedding=self._action_embedding,
                context_latents=self._latent_buffer.unsqueeze(0),
                device=self._device,
                actions=actions,
                skip_action_conditioning=self._skip_action_conditioning,
                num_inference_steps=self._num_inference_steps,
                do_classifier_free_guidance=self._do_cfg,
                guidance_scale=self._guidance_scale,
                precomputed_timesteps=self._timesteps,
                class_labels=self._class_labels,
            )

            # Sliding window in latent space (no VAE re-encode of context)
            self._latent_buffer = torch.cat(
                [self._latent_buffer[1:], new_latent], dim=0
            )
            self._action_buffer = self._action_buffer[1:] + [action_index]

            pil = decode_and_postprocess(
                vae=self._vae,
                image_processor=self._image_processor,
                latents=new_latent,
            )
        return pil


def main(model_folder: str) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = load_model(
        model_folder, device
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    batch = build_batch_from_start_image()

    img = run_inference_img_conditioning_with_params(
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        batch,
        device=device,
        skip_action_conditioning=False,
        do_classifier_free_guidance=False,
        guidance_scale=CFG_GUIDANCE_SCALE,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        image_processor=image_processor,
    )
    img.save("validation_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path to the folder containing the model weights",
    )
    args = parser.parse_args()

    main(
        model_folder=args.model_folder,
    )
