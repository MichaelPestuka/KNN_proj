# (actions, states) <-
#       ↓            |
#   Transformer      +----- next action
#       ↓            |
#   next state  -----
#       ↓
#  Segmentation Decoder
#       ↓
#   Control seg map
#       ↓
#   ControlNet
#       ↓
# Stable Diffusion
#       ↓
#   next frame


import gc
from transformer import create_mario_map
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    # StableDiffusionControlNetImg2ImgPipeline
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
    AutoencoderTiny,
    LCMScheduler,
    UniPCMultistepScheduler
)
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import DDPMScheduler
import imageio
from data_loader import encode_action, get_rollout_dataloader
import json
import time
from accelerate import Accelerator
torch.distributed
import bitsandbytes as bnb
import torchvision.transforms.functional as F_vision
from transformer import GameStateTransformer, load_transformer_checkpoints, SegmentationDecoder, LatentCompressor, GameStateTransformer, build_segmentation_target


# pip install diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0
# pip install peft
# pip install bitsandbytes

SEG_PALETTE = torch.tensor([
    [0.10, 0.10, 0.10],  # 0: background
    [0.20, 0.80, 0.20],  # 1: ground
    [0.90, 0.20, 0.20],  # 2: mario
    [0.20, 0.20, 0.90],  # 3: pipe
    [0.90, 0.80, 0.10],  # 4: enemy
    [0.80, 0.20, 0.80],  # 5: cloud
    [0.50, 0.50, 0.50],  # 6
    [0.30, 0.65, 0.35],  # 7
], dtype=torch.bfloat16)  # [8, 3]

def seg_logits_to_rgb(pred_seg: torch.Tensor) -> torch.Tensor:
    """
    pred_seg: [B, 6, H, W] logity
    returns:  [B, 3, H, W] float32 v [0,1] pro ControlNet
    """
    probs   = torch.sigmoid(pred_seg)               # [B, 6, H, W]
    palette = SEG_PALETTE.to(pred_seg.device)       # [6, 3]
    # einsum: pro každý pixel vážený součet barev přes třídy
    rgb = torch.einsum("bchw,cd->bdhw", probs, palette)
    return rgb.clamp(0.0, 1.0)

class DiffusionPipeline:
    def __init__(
        self,
        transformer,
        decoder,
        compressor,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sd_model="runwayml/stable-diffusion-v1-5",
        controlnet_model = "lllyasviel/sd-controlnet-seg",
        cache_dir="./huggingface_cache",
        torch_dtype=torch.bfloat16,
        compile=False
    ):
        self.device = device

        # --- Transformer ---
        self.transformer = transformer.to(device, dtype=torch_dtype)
        self.transformer.eval()
        
        self.decoder = decoder.to(device, dtype=torch_dtype)
        self.decoder.eval()
        
        self.compressor = compressor.to(device, dtype=torch_dtype)
        self.compressor.eval()

        # --- ControlNet ---
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )

        # --- Full pipeline ---
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        ).to(device)
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        
        # freeze SD
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        
        # Buggy but good for more FPS
        if compile:
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead", fullgraph=True)
            self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="reduce-overhead")

    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]

        array = tensor.cpu().to(torch.float32).clamp(0, 1).permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)

        return Image.fromarray(array)

    def __call__(
        self,
        states: torch.Tensor,    # [1, seq_len, total_state_dim]
        actions: torch.Tensor,   # [1, seq_len, action_dim]
        control_scale=1.0,
        num_inference_steps=4,
    ):
        assert states.shape[0] == 1, "Prozatím podpora jen pro batch_size=1"

        model_dtype = next(self.transformer.parameters()).dtype
        states  = states.to(self.device, dtype=model_dtype)
        actions = actions.to(self.device, dtype=model_dtype)

        with torch.inference_mode():
            pred     = self.transformer(states, actions)
            pred_seg = self.decoder(pred)               # [1, 6, 128, 128]

            # Mario kanál z predikované fyziky
            # pred_cx = pred["physics"][:, 4]
            # pred_cy = pred["physics"][:, 5]
            # pred_seg[:, 2:3] = create_mario_map(torch.clamp(pred_cx, 0.0, 1.0), torch.clamp(pred_cy, 0.0, 1.0), size=128) * 20.0 - 10.0

            # Segmentační mapa → RGB → upscale pro ControlNet
            control_rgb = seg_logits_to_rgb(pred_seg)   # [1, 3, 128, 128]
            control_512 = F.interpolate(
                control_rgb, size=(512, 512), mode='bilinear', align_corners=False
            )                                            # [1, 3, 512, 512]

        import torchvision
        torchvision.utils.save_image(control_512, "debug_control_image.png")


        # start = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=model_dtype):
            output = self.pipe(
                prompt="",
                image=control_512,
                height=512,
                width=512,
                guidance_scale=1.5,
                controlnet_conditioning_scale=control_scale,
                num_inference_steps=num_inference_steps,
                output_type="pt",
                return_dict=False,
                generator=torch.manual_seed(0),
            )
        # end = time.time()
        # print(f"⏱️ Inference latency {(end - start) * 1000:.2f} milliseconds")

        generated_tensor = output[0][0]
        img = (generated_tensor.cpu().to(torch.float32).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return img, pred["full"].clone().unsqueeze(1)  # [1, 1, total_state_dim]
        
class Trainer:
    def __init__(
        self,
        pipeline,
        dataloader=None,
        rollout_dataloader=None,
        accelerator=None,
        lr=1e-4,
        torch_dtype=torch.bfloat16
    ):
        self.pipeline    = pipeline
        self.dataloader  = dataloader
        self.rollout_dataloader = rollout_dataloader
        self.accelerator = accelerator
        self.device      = accelerator.device
        self.torch_dtype = torch_dtype
        self.scheduled_sampling_prob = 0.20

        self.transformer  = pipeline.transformer
        self.controlnet   = pipeline.controlnet
        self.unet         = pipeline.pipe.unet
        self.vae          = pipeline.pipe.vae
        self.decoder      = pipeline.decoder
        self.compressor   = pipeline.compressor      # nové

        self.text_encoder = pipeline.pipe.text_encoder
        self.tokenizer    = pipeline.pipe.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(pipeline.pipe.scheduler.config)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.decoder.requires_grad_(False)
        self.decoder.eval()
        self.compressor.requires_grad_(False)         # frozen — world model už natrénován
        self.compressor.eval()
        self.controlnet.requires_grad_(True)
        self.controlnet.enable_gradient_checkpointing()

        self.optimizer = bnb.optim.AdamW8bit(
            list(self.controlnet.parameters()), lr=lr
        )

        prepare_args = [self.controlnet, self.transformer, self.compressor, self.optimizer]
        if self.dataloader         is not None: prepare_args.append(self.dataloader)
        if self.rollout_dataloader is not None: prepare_args.append(self.rollout_dataloader)

        prepared = self.accelerator.prepare(*prepare_args)
        self.controlnet, self.transformer, self.compressor, self.optimizer = prepared[:4]

        idx = 4
        if self.dataloader         is not None: self.dataloader         = prepared[idx]; idx += 1
        if self.rollout_dataloader is not None: self.rollout_dataloader = prepared[idx]

        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
    
    def unfreeze_transformer(self, lr=1e-5):
        # Rozmrazíme parametry
        self.transformer.requires_grad_(True)
        self.vae.requires_grad_(False)
        
        unwrapped = self.accelerator.unwrap_model(self.transformer)
        if hasattr(unwrapped, "vae"):
            unwrapped.vae.requires_grad_(False)
            
        self.transformer.train() # Zapne zpět dropout atd.
        
        # 🚀 OPRAVA: Přidáme parametry Transformeru do existujícího optimizeru
        # Accelerate má optimizer zabalený v AcceleratedOptimizer, dostaneme se k němu přes .optimizer
        raw_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, "optimizer") else self.optimizer
        
        # Vyfiltrujeme jen ty, co opravdu chtějí grad (vynecháme zamrzlé VAE uvnitř transformeru)
        transformer_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        raw_optimizer.add_param_group({
            'params': transformer_params,
            'lr': lr
        })
        
        if self.accelerator.is_local_main_process:
            print(f"❄️➡️🔥 Transformer rozmrazen a přidán do tréninku s learning rate {lr}!")

    def unfreeze_unet(self, lr=1e-6):
        self.unet.requires_grad_(True)
        
        # Zapnutí gradient checkpointing (ošetřeno unwrappem)
        if hasattr(self.controlnet, "module"):
            self.accelerator.unwrap_model(self.controlnet).enable_gradient_checkpointing()
        else:
            self.controlnet.enable_gradient_checkpointing()
            
        self.accelerator.unwrap_model(self.unet).enable_gradient_checkpointing()

        # Dynamické sestavení parametrů (Transformer + UNet, případně ControlNet pokud už běží)
        params = list(self.transformer.parameters()) + list(self.unet.parameters())
        if next(self.controlnet.parameters()).requires_grad:
            params += list(self.controlnet.parameters())
                 
        self.optimizer = bnb.optim.AdamW8bit(params, lr=lr)

        self.unet, self.optimizer = self.accelerator.prepare(self.unet, self.optimizer)
        
        if self.accelerator.is_local_main_process:
            print(f"🔥 UNet rozmrazen a přidán do tréninku s learning rate {lr}!")
        
    def get_empty_text_embedding(self, batch_size):
        inputs = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        with torch.no_grad():
            embeddings = self.text_encoder(input_ids)[0]
        return embeddings

    def train_step_gt_pretrain(self, batch, target_idx=4):
        device = self.device
        dtype  = self.torch_dtype

        for key in ("all_segs", "all_cam_x", "all_cam_y"):
            batch[key] = batch[key].to(device)

        target        = batch["all_frames"][:, target_idx].to(device, dtype=dtype)
        target_512    = F.interpolate(target, size=(512, 512), mode='bilinear', align_corners=False)
        target_scaled = target_512 * 2.0 - 1.0
        B = target.shape[0]

        # Použij build_segmentation_target — stejná cesta jako při tréninku transformeru
        gt_seg_one_hot = build_segmentation_target(batch, target_idx)  # [B, 7, 128, 128], float, hodnoty 0/1
        gt_seg_one_hot = gt_seg_one_hot.to(device, dtype=dtype)

        # Převod na logity pro seg_logits_to_rgb (sigmoid uvnitř)
        gt_seg_logits = gt_seg_one_hot * 20.0 - 10.0                   # 0→-10, 1→+10

        control = F.interpolate(
            seg_logits_to_rgb(gt_seg_logits), size=(512, 512), mode='bilinear', align_corners=False
        )                                                               # [B, 3, 512, 512]

        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            with torch.no_grad():
                latents = self.vae.encode(target_scaled).latent_dist.sample() * 0.18215

            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                        (B,), device=device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = self.get_empty_text_embedding(B).to(dtype=dtype)

            down_samples, mid_sample = self.controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control, return_dict=False
            )
            noise_pred = self.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample
            ).sample

            loss = F.mse_loss(noise_pred, noise)

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    def train_step(self, batch):
        device = self.device
        dtype  = self.torch_dtype

        for key in ("all_segs", "all_cam_x", "all_cam_y"):
            batch[key] = batch[key].to(device)

        # Stavová sekvence z GT snímků
        unwrapped_t = self.accelerator.unwrap_model(self.transformer)
        unwrapped_c = self.accelerator.unwrap_model(self.compressor)

        frames = batch["all_frames"][:, :4].to(device, dtype=dtype)
        B, T, C, H, W = frames.shape
        with torch.no_grad():
            z      = unwrapped_t.encode_to_latent(frames.reshape(B * T, C, H, W))
            visual = unwrapped_c(z).view(B, T, -1)

        x  = batch["all_x"][:,  :4].to(device)
        y  = batch["all_y"][:,  :4].to(device)
        vx = batch["all_vx"][:, :4].to(device)
        vy = batch["all_vy"][:, :4].to(device)
        cx = batch["all_cam_x"][:, :4]
        cy = batch["all_cam_y"][:, :4]
        physics      = torch.stack([x, y, vx, vy, cx, cy], dim=-1)
        latent_state = unwrapped_t.init_latent.expand(B, T, -1).to(device=device, dtype=dtype)
        states  = torch.cat([physics, visual, latent_state], dim=-1).to(dtype=dtype)
        actions = batch["all_actions"][:, :4].to(device, dtype=dtype)

        # Scheduled sampling — přidej šum na část kontextových snímků
        if torch.rand(1).item() < self.scheduled_sampling_prob:
            noisy = frames.clone()
            for i in range(T):
                if torch.rand(1).item() < 0.5:
                    noisy[:, i] = torch.clamp(
                        noisy[:, i] + torch.randn_like(noisy[:, i]) * 0.15, 0.0, 1.0
                    )
            with torch.no_grad():
                z_n      = unwrapped_t.encode_to_latent(noisy.reshape(B * T, C, H, W))
                visual_n = unwrapped_c(z_n).view(B, T, -1)
            states = torch.cat([physics, visual_n, latent_state], dim=-1).to(dtype=dtype)

        # Target snímek
        target = batch["all_frames"][:, 4].to(device, dtype=dtype)
        target_512    = F.interpolate(target, size=(512, 512), mode='bilinear', align_corners=False)
        target_scaled = target_512 * 2.0 - 1.0

        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            with torch.no_grad():
                latents = self.vae.encode(target_scaled).latent_dist.sample() * 0.18215

            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                        (B,), device=device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Transformer → seg mapa → RGB control
            pred     = self.transformer(states, actions)
            pred_seg = self.decoder(pred)
            # pred_seg[:, 2:3] = (
            #     create_mario_map(torch.clamp(pred["physics"][:, 4], 0.0, 1.0), torch.clamp(pred["physics"][:, 5], 0.0, 1.0), size=128)
            #     * 20.0 - 10.0
            # )
            control = F.interpolate(
                seg_logits_to_rgb(pred_seg), size=(512, 512), mode='bilinear', align_corners=False
            )

            encoder_hidden_states = self.get_empty_text_embedding(B).to(dtype=dtype)
            down_samples, mid_sample = self.controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control, return_dict=False
            )
            noise_pred = self.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample
            ).sample

            loss = F.mse_loss(noise_pred, noise)

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    def train_step_rollout(self, batch, seq_len=4, rollout_steps=10):
        device = self.device
        dtype  = self.torch_dtype

        for key in ("all_segs", "all_cam_x", "all_cam_y"):
            batch[key] = batch[key].to(device)

        unwrapped_t = self.accelerator.unwrap_model(self.transformer)
        unwrapped_c = self.accelerator.unwrap_model(self.compressor)

        # Inicializace stavu z GT
        frames = batch["all_frames"][:, :seq_len].to(device, dtype=dtype)
        B, T, C, H, W = frames.shape
        with torch.no_grad():
            z      = unwrapped_t.encode_to_latent(frames.reshape(B * T, C, H, W))
            visual = unwrapped_c(z).view(B, T, -1)

        x  = batch["all_x"][:,  :seq_len].to(device)
        y  = batch["all_y"][:,  :seq_len].to(device)
        vx = batch["all_vx"][:, :seq_len].to(device)
        vy = batch["all_vy"][:, :seq_len].to(device)
        cx = batch["all_cam_x"][:, :seq_len]
        cy = batch["all_cam_y"][:, :seq_len]
        physics      = torch.stack([x, y, vx, vy, cx, cy], dim=-1)
        latent_state = unwrapped_t.init_latent.expand(B, T, -1).to(device=device, dtype=dtype)
        states = torch.cat([physics, visual, latent_state], dim=-1).to(dtype=dtype)

        total_batch_loss = 0.0
        self.optimizer.zero_grad()

        for step in range(rollout_steps):
            t_target = seq_len + step
            actions  = batch["all_actions"][:, step:step + seq_len].to(device, dtype=dtype)

            with self.accelerator.autocast():
                pred     = self.transformer(states, actions)
                pred_seg = self.decoder(pred)
                # pred_seg[:, 2:3] = (
                #     create_mario_map(torch.clamp(pred["physics"][:, 4], 0.0, 1.0), torch.clamp(pred["physics"][:, 5], 0.0, 1.0), size=128)
                #     * 20.0 - 10.0
                # )
                control = F.interpolate(
                    seg_logits_to_rgb(pred_seg), size=(512, 512),
                    mode='bilinear', align_corners=False
                )

                target = batch["all_frames"][:, t_target].to(device, dtype=dtype)
                target_scaled = F.interpolate(
                    target, size=(512, 512), mode='bilinear', align_corners=False
                ) * 2.0 - 1.0

                with torch.no_grad():
                    latents = self.vae.encode(target_scaled).latent_dist.sample() * 0.18215

                noise         = torch.randn_like(latents)
                timesteps     = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                            (B,), device=device).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = self.get_empty_text_embedding(B).to(dtype=dtype)
                down_samples, mid_sample = self.controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control, return_dict=False
                )
                noise_pred = self.unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                ).sample

                step_loss = F.mse_loss(noise_pred, noise) / rollout_steps

            self.accelerator.backward(step_loss)
            total_batch_loss += step_loss.item()

            # Autoregresivní update stavu — detach aby se BPTT neakumulovalo
            if step < rollout_steps - 1:
                next_state = pred["full"].detach().unsqueeze(1)  # [B, 1, total_state_dim]

                states = torch.cat([states[:, 1:], next_state], dim=1)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        self.optimizer.step()

        return total_batch_loss
    
    def train(self, epochs=10, save_dir="./checkpoints", save_every=2, video_every=2, phase="standard", rollout_steps=10):
        """
        phase: 'gt_pretrain' | 'standard' | 'rollout'
        """
        torch.cuda.empty_cache()
        
        current_dataloader = self.rollout_dataloader

        for epoch in range(epochs):
            if self.accelerator.is_local_main_process:
                loop = tqdm(current_dataloader)
                loop.set_description(f"Epoch {epoch} [{phase.upper()}]")
            else:
                loop = current_dataloader

            for batch in loop:
                if phase == "gt_pretrain":
                    loss = self.train_step_gt_pretrain(batch)
                elif phase == "standard":
                    loss = self.train_step(batch)
                elif phase == "rollout":
                    loss = self.train_step_rollout(
                        batch, 
                        rollout_steps=rollout_steps
                    )
                else:
                    raise ValueError(f"Neznámá fáze tréninku: {phase}")
                    
                if self.accelerator.is_local_main_process:
                    loop.set_postfix(loss=loss)
            
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_local_main_process and (epoch + 1) % save_every == 0:
                self.save_checkpoint(output_dir=save_dir, epoch=epoch)
                
            if self.accelerator.is_local_main_process and (epoch + 1) % video_every == 0:
                os.makedirs("./videos", exist_ok=True)
                generate_video(
                    self.pipeline,
                    episode_path="../data-generation/super-mario-bros/collected_data/combined_w1s1_00896dbf",
                    output_path=f"{save_dir}/epoch_{epoch}_{phase}.mp4",
                    num_frames=300,
                    image_size=256,
                    fps=30
                )
            
            # Zvyšování sampling prob pouze ve standardní fázi
            if phase == "standard":
                self.scheduled_sampling_prob = min(0.85, self.scheduled_sampling_prob + 0.1) 
                
            self.accelerator.wait_for_everyone()
                
    def save_checkpoint(self, output_dir="./checkpoints", epoch=0):
        os.makedirs(output_dir, exist_ok=True)
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        unwrapped_controlnet = self.accelerator.unwrap_model(self.controlnet)

        controlnet_path = os.path.join(epoch_dir, "controlnet")
        unwrapped_controlnet.save_pretrained(controlnet_path)

        print(f"✅ Controlnet checkpoint successfully saved to: {epoch_dir}\n")
        
def generate_video(
    pipeline,
    episode_path,
    output_path="video.mp4",
    num_frames=200,
    image_size=256,
    fps=30,
    num_inference_steps=4
):
    json_file = [f for f in os.listdir(episode_path) if f.endswith(".json")][0]
    with open(os.path.join(episode_path, json_file), 'r') as f:
        data = json.load(f)
        
    frames_meta = data["frames"]
    device = pipeline.device
    
    # 🚀 OPRAVA 1: Transformer má pouze seq_len, který platí pro oboje (akce i stavy)
    seq_len = pipeline.transformer.seq_len
    dtype = next(pipeline.transformer.parameters()).dtype

    print("📦 Loading episode actions and the INITIAL frame...")
    all_actions = []
    
    # 1. Načíst VŠECHNY akce z epizody
    for meta in frames_meta:
        action_tensor = encode_action(meta["action"]).to(device, dtype=dtype)
        all_actions.append(action_tensor)

    # 2. Načíst POUZE PRVNÍ snímek pro inicializaci
    first_meta = frames_meta[0]
    fname = first_meta["filename"]
    
    img_path = os.path.join(episode_path, first_meta["filename"])
    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device, dtype=dtype)
    
    # 🚀 OPRAVA 2: Vytvoření GT stavu z prvního snímku
    with torch.no_grad():
        z_first = pipeline.transformer.encode_to_latent(img_tensor)
        visual_first = pipeline.compressor(z_first).view(1, 1, -1)
        
    WORLD_W = 3400.0
    VIEWPORT_H = 240.0
    
    x = 40 / WORLD_W
    y = 79 / VIEWPORT_H
    vx, vy = 0.0, 0.0
    cx, cy = 0.36, 0.849  # GT segmentaci tu nemáme, začínáme bezpečně na středu kamery    
    
    physics_first = torch.tensor([[[x, y, vx, vy, cx, cy]]], device=device, dtype=dtype)
    latent_first = pipeline.transformer.init_latent.clone().to(device=device, dtype=dtype)
    
    state_first = torch.cat([physics_first, visual_first, latent_first], dim=-1) # [1, 1, total_state_dim]
    
    # Historie stavů inicializovaná nulami. Poslední krok bude první validní stav.
    states = torch.zeros((1, seq_len, pipeline.transformer.total_state_dim), device=device, dtype=dtype)
    
    latent_dim = pipeline.transformer.latent_state_dim
    states[:, :, -latent_dim:] = pipeline.transformer.init_latent.expand(1, seq_len, -1)
    
    states[:, -1:] = state_first
    
    # Pro video si první snímek zvětšíme na 512x512
    img_for_vid = img.resize((512, 512), Image.Resampling.LANCZOS)
    images_for_video = [np.array(img_for_vid)]

    noop_action = torch.tensor([0., 0., 0., 0., 1.], device=device, dtype=dtype)

    print("🚀 Generating video autoregressively...")

    gc.collect()
    torch.cuda.empty_cache()
    
    max_steps = min(len(frames_meta) - 1, num_frames)

    # --- 🔒 BEZPEČNOSTNÍ BLOK: Uložení stavů a přepnutí do eval ---
    was_transformer_training = pipeline.transformer.training
    was_controlnet_training = pipeline.controlnet.training
    was_unet_training = pipeline.pipe.unet.training

    pipeline.transformer.eval()
    pipeline.controlnet.eval()
    pipeline.pipe.unet.eval()

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in tqdm(range(max_steps), desc="Generating", unit=" frame"):

                # --- 1. PŘÍPRAVA HISTORIE AKCÍ ---
                start_idx = max(0, t - seq_len + 1)
                recent_actions = all_actions[start_idx : t + 1]

                if len(recent_actions) < seq_len:
                    pad_actions = [noop_action] * (seq_len - len(recent_actions))
                    recent_actions = pad_actions + recent_actions

                action_stack = torch.stack(recent_actions).unsqueeze(0)
                
                # --- 2. INFERENCE ---
                # Nyní správně předáváme tensor `states` místo obrázků
                next_frame_np, next_state = pipeline(
                    states, 
                    action_stack, 
                    num_inference_steps=num_inference_steps
                )
                
                images_for_video.append(next_frame_np)
                
                # --- 3. AUTOREGRESIVNÍ POSUN ---
                states = torch.cat([states[:, 1:], next_state], dim=1)
                
    except Exception as e:
        print(f"❌ Error during generation at step {t}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # --- 🧹 ÚKLID: Návrat do původního stavu ---
        if was_transformer_training:
            pipeline.transformer.train()
        if was_controlnet_training:
            pipeline.controlnet.train()
        if was_unet_training:
            pipeline.pipe.unet.train()

    print(f"💾 Video saved to {output_path}...")
    imageio.mimsave(output_path, images_for_video, fps=fps)
        
def load_controlnet_checkpoint(pipeline, output_dir="./checkpoints", epoch=0, device="cuda", unet=False):
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        target_dtype = pipeline.pipe.unet.dtype

        controlnet_path = os.path.join(epoch_dir, "controlnet")
        if os.path.exists(controlnet_path):
            pipeline.controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=target_dtype
            )
            pipeline.controlnet.to(device)
            pipeline.pipe.controlnet = pipeline.controlnet
            print("✅ ControlNet úspěšně načten.")
        else:
            print("⚠️ ControlNet nebyl v tomto checkpointu nalezen.")
            
        print(f"✅ Loading controlnet epoch {epoch} checkpoint completed.")
        return True
    
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
        
    accelerator = Accelerator(mixed_precision="bf16")

    # Vypisovat informace chceme jen na hlavním procesu (GPU 0), aby se nezdvojovaly logy
    if accelerator.is_local_main_process:
        print(f"🚀 Spouštím trénink na {accelerator.num_processes} GPU!")
        
    img_size = 256
    seq_len = 4
        
    data_loader_rollout = get_rollout_dataloader(
        root_dir="../data-generation/super-mario-bros/collected_data",
        seg_dir="mario_data_seg",
        batch_size=64,
        shuffle=True,
        num_workers=4,
        image_size=256,
        num_actions=seq_len,
        num_frames=seq_len,
        rollout_steps=20,
        stride=20,
        max_episodes=200
    )
    
    transformer = GameStateTransformer()
    decoder = SegmentationDecoder()
    compressor = LatentCompressor()
    
    # --- Init pipeline ---
    pipeline = DiffusionPipeline(transformer, decoder, compressor, device=accelerator.device, torch_dtype=torch.bfloat16, compile=False)

    # load_controlnet_checkpoint(pipeline, output_dir="./checkpoints_controlnet", epoch=9, device=accelerator.device, unet=False)
    
    load_transformer_checkpoints(
        "checkpoints_transformer/checkpoint_rollout.pt",
        pipeline.transformer,
        pipeline.decoder,
        pipeline.compressor,
        accelerator=None,
        device="cpu"
    )

    trainer = Trainer(pipeline, dataloader=None, rollout_dataloader=data_loader_rollout, accelerator=accelerator, lr=1e-5)
        
    trainer.train(epochs=14, save_dir="./checkpoints", phase="gt_pretrain", save_every=13, video_every=4)
    
    trainer.train(epochs=8, save_dir="./checkpoints_controlnet", phase="standard", save_every=5, video_every=2)
    
    trainer.train(epochs=7, save_dir="./checkpoints_controlnet_rollout", phase="rollout", save_every=4, video_every=2, rollout_steps=5)
    
    trainer.train(epochs=6, save_dir="./checkpoints_controlnet_rollout", phase="rollout", save_every=4, video_every=2, rollout_steps=10)
    
    trainer.train(epochs=5, save_dir="./checkpoints_controlnet_rollout", phase="rollout", save_every=4, video_every=2, rollout_steps=15)
    
    trainer.train(epochs=10, save_dir="./checkpoints_controlnet_rollout", phase="rollout", save_every=4, video_every=2, rollout_steps=20)
    
    
    # trainer.unfreeze_transformer(lr=1e-5)
    
    # trainer.train(epochs=4, save_dir="./checkpoints_std", save_every=2, video_every=1, is_rollout_phase=False)
    
    # trainer.train(epochs=10, save_dir="./checkpoints_rollout", save_every=2, video_every=1, is_rollout_phase=True, rollout_steps=10)
    
    # trainer.unfreeze_unet(lr=1e-6)
