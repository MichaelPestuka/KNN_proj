import torch
import numpy as np
from PIL import Image
from transformer import GameStateTransformer, SegmentationDecoder, LatentCompressor, load_transformer_checkpoints
import os
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from pipeline import seg_logits_to_rgb
import torch.nn.functional as F
import time
import imageio

class InferenceEngine:
    """
    Vysoce optimalizovaný engine pro real-time inferenci.
    Udržuje si interní historii stavů a akcí, takže zvenčí vyžaduje
    pouze aktuální akci a vrací vygenerovaný snímek.
    """
    def __init__(
        self,
        transformer_path,
        controlnet_path,
        seq_len=4,
        num_inference_steps=4,
        guidance_scale=30.0,
        control_scale=1.3,
        seed=42,
        device="cuda",
        target_dtype=torch.float16,
        cache_dir="./huggingface_cache",
        sd_model="runwayml/stable-diffusion-v1-5",
        compile=False
    ):
        
        self.device = device
        self.dtype = target_dtype
        
        self.seq_len = seq_len
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.control_scale = control_scale
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # Výchozí NO-OP akce (uprav podle své definice akcí)
        self.noop_action = torch.tensor([0., 0., 0., 0., 1.], device=self.device, dtype=self.dtype)

        # Interní buffery pro historii
        self.states_buffer = None
        self.actions_buffer = None
        
        self.transformer = GameStateTransformer().to(device, dtype=target_dtype)
        self.decoder = SegmentationDecoder().to(device, dtype=target_dtype)
        self.compressor = LatentCompressor().to(device, dtype=target_dtype)
        
        # --- ControlNet ---
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=target_dtype,
            cache_dir=cache_dir
        )

        # --- Full pipeline ---
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=self.controlnet,
            torch_dtype=target_dtype,
            cache_dir=cache_dir
        ).to(device)
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        
        load_transformer_checkpoints(
            transformer_path,
            self.transformer,
            self.decoder,
            self.compressor,
            accelerator=None,
            device="cpu"
        )

        # Přepnutí všech komponent do evaluačního módu pro jistotu
        self.transformer.eval()
        self.controlnet.eval()
        self.pipe.unet.eval()
        self.decoder.eval()
        self.compressor.eval()

        if compile:
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead", fullgraph=True)
            self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
            self.transformer = torch.compile(self.transformer, mode="reduce-overhead", fullgraph=True)
            self.decoder = torch.compile(self.decoder, mode="reduce-overhead", fullgraph=True)
            self.compressor = torch.compile(self.compressor, mode="reduce-overhead", fullgraph=True)

    @torch.inference_mode()
    def reset(self, initial_image_path, initial_physics=None):
        """
        Inicializuje stavovou historii z prvního snímku.
        initial_image_tensor: [1, 3, H, W] tensor v rozsahu 0.0 - 1.0
        initial_physics: [1, 1, 6] tensor fyziky (x, y, vx, vy, cx, cy)
        """
        # 1. Zpracování prvního snímku do latentního/vizuálního stavu
        img = Image.open(initial_image_path).convert("RGB").resize((256, 256))
    
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        initial_image_tensor = img_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)
        z_first = self.transformer.encode_to_latent(initial_image_tensor)
        visual_first = self.compressor(z_first).view(1, 1, -1)

        # 2. Příprava fyzikálního stavu
        if initial_physics is None:
            WORLD_W, VIEWPORT_H = 3400.0, 240.0
            x, y = 40 / WORLD_W, 79 / VIEWPORT_H
            vx, vy, cx, cy = 0.0, 0.0, 0.36, 0.849
            initial_physics = torch.tensor([[[x, y, vx, vy, cx, cy]]], device=self.device, dtype=self.dtype)
        else:
            initial_physics = initial_physics.to(self.device, dtype=self.dtype)

        # 3. Inicializace skrytého latentního stavu
        latent_first = self.transformer.init_latent.clone().to(self.device, dtype=self.dtype)

        # 4. Složení plného prvního stavu
        state_first = torch.cat([initial_physics, visual_first, latent_first], dim=-1) # [1, 1, total_state_dim]

        # 5. Inicializace bufferů historií
        total_state_dim = self.transformer.total_state_dim
        self.states_buffer = torch.zeros((1, self.seq_len, total_state_dim), device=self.device, dtype=self.dtype)
        
        # Předvyplnění latentních dimenzí inicializačním vektorem
        latent_dim = self.transformer.latent_state_dim
        self.states_buffer[:, :, -latent_dim:] = self.transformer.init_latent.expand(1, self.seq_len, -1)
        
        # Poslední prvek historie nastavíme na náš první reálný stav
        self.states_buffer[:, -1:] = state_first

        # Inicializace historie akcí (vyplníme NOOP)
        self.actions_buffer = [self.noop_action.clone() for _ in range(self.seq_len)]

        print("Inference Engine resetován a připraven.")

    @torch.inference_mode()
    def step(self, action_tensor):
        """
        Provede jeden krok generování.
        action_tensor: [action_dim] tensor s aktuální akcí, např. [0., 0., 0., 0., 1.] - ["left", "right", "A", "B", "NOOP"]
        Vrací: numpy array (RGB obrázek) např. pro odeslání přes WebSocket / zobrazení
        """
        if self.states_buffer is None:
            raise RuntimeError("Engine není inicializován! Zavolej nejdřív reset(initial_image).")
        
        torch.compiler.cudagraph_mark_step_begin()

        action_tensor = action_tensor.to(self.device, dtype=self.dtype)

        # 1. Aktualizace historie akcí (FIFO: odstraníme nejstarší, přidáme novou)
        self.actions_buffer.pop(0)
        self.actions_buffer.append(action_tensor)
        
        # [1, seq_len, action_dim]
        action_stack = torch.stack(self.actions_buffer).unsqueeze(0)

        # 2. Samotná inference
        with torch.autocast(device_type="cuda", dtype=self.dtype):            
            pred     = self.transformer(self.states_buffer, action_stack)
            next_state = pred["full"].clone().unsqueeze(1)
            
            pred_seg = self.decoder(pred)

            control_rgb = seg_logits_to_rgb(pred_seg)   # [1, 3, 128, 128]
            control_512 = F.interpolate(
                control_rgb, size=(512, 512), mode='bilinear', align_corners=False
            )
            
            output = self.pipe(
                prompt="",
                image=control_512,
                height=512,
                width=512,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.control_scale,
                num_inference_steps=self.num_inference_steps,
                output_type="pt",
                return_dict=False,
                generator=self.generator,
            )

        # 3. Autoregresivní posun historie stavů
        self.states_buffer = torch.cat([self.states_buffer[:, 1:], next_state], dim=1)

        # 4. Vrácení vygenerovaného snímku
        generated_tensor = output[0][0]
        img = (generated_tensor.cpu().to(torch.float32).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return img
        
def main():
    engine = InferenceEngine(
        transformer_path="./inference_checkpoints/transformer.pt",
        controlnet_path="./inference_checkpoints/controlnet",
        num_inference_steps=4,
        guidance_scale=30.0,
        control_scale=1.3,
        device="cuda",
        target_dtype=torch.float16,
        compile=True
    )
    
    engine.reset(initial_image_path="./initial_frame.jpg")
    
    imgs = []
    for _ in range(100):
        action = torch.tensor([1., 0., 1., 0., 0.], device=engine.device, dtype=engine.dtype)
        
        start = time.time()
        img = engine.step(action)
        
        imgs.append(img)
        
        end = time.time()
        print(f"inference time: {end - start:.4f} s")
    imageio.mimsave("./generated_video.gif", imgs, fps=4)


if __name__ == "__main__":
    main()