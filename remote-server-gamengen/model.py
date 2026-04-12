import json
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from config import NUM_BUCKETS, PRETRAINED_MODEL_NAME_OR_PATH


def load_embedding_info_dict(model_folder: str) -> dict:
    if os.path.exists(model_folder):
        with open(os.path.join(model_folder, "embedding_info.json"), "r") as f:
            embedding_info = json.load(f)
    else:
        file_path = hf_hub_download(
            repo_id=model_folder, filename="embedding_info.json", repo_type="model"
        )
        with open(file_path, "r") as f:
            embedding_info = json.load(f)
    return embedding_info


def load_action_embedding(
    model_folder: str, action_num_embeddings: int
) -> torch.nn.Embedding:
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_num_embeddings, embedding_dim=768
    )
    if os.path.exists(model_folder):
        action_embedding.load_state_dict(
            load_file(os.path.join(model_folder, "action_embedding_model.safetensors"))
        )
    else:
        file_path = hf_hub_download(
            repo_id=model_folder,
            filename="action_embedding_model.safetensors",
            repo_type="model",
        )
        action_embedding.load_state_dict(load_file(file_path))
    return action_embedding


def load_model(
    model_folder: str, device: torch.device | None = None
) -> tuple[
    UNet2DConditionModel,
    AutoencoderKL,
    torch.nn.Embedding,
    DDIMScheduler,
    CLIPTokenizer,
    CLIPTextModel,
]:
    """
    Load a model from the hub

    Args:
        model_folder: the folder to load the model from, can be a model id or a local folder
    """
    embedding_info = load_embedding_info_dict(model_folder)
    action_embedding = load_action_embedding(
        model_folder=model_folder,
        action_num_embeddings=embedding_info["num_embeddings"],
    )

    noise_scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="noise_scheduler"
    )

    vae = AutoencoderKL.from_pretrained(model_folder, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_folder, subfolder="unet")

    assert (
        noise_scheduler.config.prediction_type == "v_prediction"
    ), "Noise scheduler prediction type should be 'v_prediction'"
    assert (
        unet.config.num_class_embeds == NUM_BUCKETS
    ), f"UNet num_class_embeds should be {NUM_BUCKETS}"

    # Unaltered
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )

    if device:
        unet = unet.to(device)
        vae = vae.to(device)
        action_embedding = action_embedding.to(device)
        text_encoder = text_encoder.to(device)
        # FP16 UNet/VAE/embedding on CUDA for faster inference (Blackwell and similar).
        if device.type == "cuda":
            unet = unet.to(dtype=torch.float16)
            vae = vae.to(dtype=torch.float16)
            action_embedding = action_embedding.to(dtype=torch.float16)

    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder
