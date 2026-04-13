from json import encoder
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from data_loader import get_dataloader, get_rollout_dataloader
from accelerate import Accelerator
from diffusers import AutoencoderTiny
        
class GameStateTransformer(nn.Module):
    def __init__(
        self,
        visual_embed_dim=124,
        action_dim=5,
        d_model=256,
        nhead=8,
        num_layers=4,
        seq_len=4,
        latent_state_dim=64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.physics_dim = 6
        self.visual_embed_dim = visual_embed_dim
        self.latent_state_dim = latent_state_dim
        self.total_state_dim = self.physics_dim + visual_embed_dim + latent_state_dim

        self.action_embed = nn.Linear(action_dim, 32)
        self.input_proj   = nn.Linear(self.total_state_dim + 32, d_model)

        self.pos_emb   = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="../huggingface_cache"
        )
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.physics_head      = nn.Linear(d_model, self.physics_dim)
        self.visual_head       = nn.Linear(d_model, visual_embed_dim)   # Varianta A
        self.latent_state_head = nn.Sequential(
            nn.Linear(d_model, latent_state_dim),
            nn.LayerNorm(latent_state_dim)
        )
        
        self.init_latent = nn.Parameter(torch.randn(1, 1, latent_state_dim) * 0.02)

    def forward(self, states, actions):
        # states:  [B, T, total_state_dim]
        # actions: [B, T, action_dim]
        B, T, _ = states.shape

        # Vytáhneme si absolutní hodnoty z úplně posledního známého stavu
        # states[:, -1] má tvar [B, total_state_dim]
        last_state = states[:, -1]
        last_physics = last_state[:, :self.physics_dim]
        last_visual  = last_state[:, self.physics_dim : self.physics_dim + self.visual_embed_dim]

        action_emb = self.action_embed(actions)               # [B, T, 32]
        x = torch.cat([states, action_emb], dim=-1)
        x = self.input_proj(x)                                # [B, T, d_model]
        x = x + self.pos_emb[:, :T]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                        # [B, T+1, d_model]

        out     = self.transformer(x)
        cls_out = out[:, 0]                                    # [B, d_model]

        # Residual connections (predikujeme deltu a rovnou ji přičteme)
        physics_next      = last_physics + self.physics_head(cls_out)   # [B, 6]
        visual_next       = last_visual  + self.visual_head(cls_out)    # [B, visual_embed_dim]
        
        latent_state_next = self.latent_state_head(cls_out)             # [B, latent_state_dim]

        return {
            "physics":      physics_next,
            "visual":       visual_next,
            "latent_state": latent_state_next,
            "full": torch.cat([physics_next, visual_next, latent_state_next], dim=-1),
        }

    def encode_to_latent(self, image_batch):
        return self.vae.encode(image_batch * 2.0 - 1.0).latents

    def decode_to_image(self, latent_batch):
        return torch.clamp(self.vae.decode(latent_batch).sample / 2.0 + 0.5, 0.0, 1.0)
    
class LatentCompressor(nn.Module):
    def __init__(self, latent_dim=124):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), # 8×8
            nn.ReLU(),
        )
        self.proj = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, z):
        x = self.encoder(z)           # [B, 32, 8, 8]
        return self.proj(x.flatten(1))  # [B, latent_dim]
        
class SegmentationDecoder(nn.Module):
    def __init__(self, visual_embed_dim=124, latent_state_dim=64, num_classes=8):
        super().__init__()
        in_dim = visual_embed_dim + latent_state_dim
        self.pre = nn.Linear(in_dim, 64 * 8 * 8)  # více kanálů na startu

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),   # GroupNorm > BatchNorm pro malé batch
                nn.GELU(),
                # Residual refinement po upsample
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
            )

        self.up1 = up_block(64, 64)   # 8→16
        self.up2 = up_block(64, 32)   # 16→32
        self.up3 = up_block(32, 32)   # 32→64
        self.up4 = up_block(32, 16)   # 64→128

        # Auxiliary head ze 32×32 — intermediate supervision
        self.aux_head = nn.Conv2d(32, num_classes, 1)
        # Hlavní výstup
        self.final    = nn.Conv2d(16, num_classes, 1)

    def forward(self, pred: dict):
        z = torch.cat([pred["visual"], pred["latent_state"]], dim=-1)
        x = self.pre(z).view(-1, 64, 8, 8)

        x = self.up1(x)   # [B, 64, 16, 16]
        x = self.up2(x)   # [B, 32, 32, 32]
        aux = self.aux_head(x)   # [B, C, 32, 32] — auxiliary output
        x = self.up3(x)   # [B, 32, 64, 64]
        x = self.up4(x)   # [B, 16, 128, 128]

        return self.final(x), aux  # vrátí tuple
    
def boundary_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spočítá BCE pouze na pixelech u hranic tříd v targetu.
    pred_logits: [B, C, H, W]
    target:      [B, C, H, W] one-hot float
    """
    # Sobel na target one-hot → kde jsou hrany
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], 
                             dtype=target.dtype, device=target.device)
    sobel_y = sobel_x.T

    # Aplikuj na každý kanál zvlášť
    B, C, H, W = target.shape
    t_flat = target.view(B * C, 1, H, W)
    
    kx = sobel_x.view(1, 1, 3, 3)
    ky = sobel_y.view(1, 1, 3, 3)
    
    edge_x = F.conv2d(t_flat, kx, padding=1)
    edge_y = F.conv2d(t_flat, ky, padding=1)
    edge_mag = (edge_x**2 + edge_y**2).sqrt().view(B, C, H, W)
    
    # Normalizovaná maska hran [0, 1]
    edge_mask = (edge_mag > 0.3).float()
    
    if edge_mask.sum() < 1:
        return torch.tensor(0.0, device=target.device)

    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    return (bce * edge_mask).sum() / edge_mask.sum().clamp(min=1)

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, 
              eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice loss přes všechny kanály.
    Ideální pro malé objekty (Mario, enemy) — BCE je vůči nim slepá.
    """
    pred = torch.sigmoid(pred_logits)    # [B, C, H, W]
    
    # Flatten spatial dims
    pred_f   = pred.flatten(2)           # [B, C, H*W]
    target_f = target.flatten(2)         # [B, C, H*W]
    
    intersection = (pred_f * target_f).sum(dim=2)          # [B, C]
    union        = pred_f.sum(dim=2) + target_f.sum(dim=2) # [B, C]
    
    dice = (2 * intersection + eps) / (union + eps)        # [B, C]
    return 1.0 - dice.mean()
    
def create_mario_map(cam_x, cam_y, size=128, box_h=0.062, box_w=0.05):
    B      = cam_x.shape[0]
    device = cam_x.device
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing='ij'
    )
    cx = cam_x.view(B, 1, 1)
    cy = cam_y.view(B, 1, 1)
    in_box = (
        (grid_x.unsqueeze(0) >= cx - box_w / 2) &
        (grid_x.unsqueeze(0) <= cx + box_w / 2) &
        (grid_y.unsqueeze(0) >= cy - box_h / 2) &
        (grid_y.unsqueeze(0) <= cy + box_h / 2)
    )
    return in_box.float().unsqueeze(1)

def build_state_sequence(batch, seq_len, model, compressor, device):
    frames = batch["all_frames"][:, :seq_len].to(device)
    B, T, C, H, W = frames.shape
    unwrapped_model = accelerator.unwrap_model(model)
    with torch.no_grad():
        z      = unwrapped_model.encode_to_latent(frames.reshape(B * T, C, H, W))  # [B*T, 4, 32, 32]
    visual = compressor(z).view(B, T, -1)                              # [B, T, visual_embed_dim]

    x  = batch["all_x"][:,  :seq_len].to(device)
    y  = batch["all_y"][:,  :seq_len].to(device)
    vx = batch["all_vx"][:, :seq_len].to(device)
    vy = batch["all_vy"][:, :seq_len].to(device)
    cx = batch["all_cam_x"][:, :seq_len].to(device)
    cy = batch["all_cam_y"][:, :seq_len].to(device)
    physics = torch.stack([x, y, vx, vy, cx, cy], dim=-1)             # [B, T, 6]

    latent_state = unwrapped_model.init_latent.expand(B, T, -1).to(device)

    return torch.cat([physics, visual, latent_state], dim=-1)          # [B, T, total_state_dim]

def build_target_state(batch, t):
    x = batch["all_x"][:, t]
    y = batch["all_y"][:, t]
    vx = batch["all_vx"][:, t]
    vy = batch["all_vy"][:, t]
    cx = batch["all_cam_x"][:, t]
    cy = batch["all_cam_y"][:, t]

    return torch.stack([x, y, vx, vy, cx, cy], dim=-1)

def build_segmentation_target(batch, t, num_classes=8):
    seg_idx = batch["all_segs"][:, t].long()   # [B, H, W]
    B, H, W = seg_idx.shape
    device  = seg_idx.device

    seg_small = F.interpolate(
        seg_idx.unsqueeze(1).float(), size=(128, 128), mode="nearest"
    ).squeeze(1).long()                          # [B, 128, 128]

    # vynuluj class 2 z color matchingu (nespolehlivé) — přepíše se gaussiánem
    seg_small[seg_small == 2] = 0

    one_hot = torch.zeros(B, num_classes, 128, 128, device=device)
    one_hot.scatter_(1, seg_small.unsqueeze(1), 1.0)

    # Mario kanál z camera-space centroidu
    cam_x = batch["all_cam_x"][:, t].to(device)  # [B], normalizované [0,1]
    cam_y = batch["all_cam_y"][:, t].to(device)  # [B], normalizované [0,1]
    one_hot[:, 2] = create_mario_map(torch.clamp(cam_x, 0.0, 1.0), torch.clamp(cam_y, 0.0, 1.0), size=128).squeeze(1)

    return one_hot  # [B, 6, 128, 128]

def save_segmentation_debug(pred, target, path, target_image):
    pred = torch.sigmoid(pred)

    pred_img = pred[0]
    target_seg = target[0]
    target_image = target_image[0]
    
    # Definice palety barev pro 7 tříd (RGB formát).
    # Hodnoty lze libovolně upravit podle toho, co třídy skutečně reprezentují.
    colors = torch.tensor([
        [0.2, 0.2, 0.2],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ], device=pred_img.device).view(8, 3, 1, 1)

    # Vynásobíme predikční kanály [6, H, W] s paletou [6, 3, 1, 1]
    # a sečteme je přes dimenzi tříd. Získáme tak RGB obrázek [3, H, W].
    # Funkce clamp(0, 1) zajistí, že barvy nepřetečou, pokud by se třídy překrývaly.
    pred_rgb = (pred_img.unsqueeze(1) * colors).sum(dim=0).clamp(0, 1)
    target_rgb = (target_seg.unsqueeze(1) * colors).sum(dim=0).clamp(0, 1)

    # Upscale pred a target na 256x256 pro lepší vizualizaci
    pred_rgb = F.interpolate(
        pred_rgb.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    target_rgb = F.interpolate(
        target_rgb.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
    ).squeeze(0)

    # Spojení do jedné vizualizace (Target RGB | Target Seg | Pred Seg)
    combined = torch.cat([target_image, target_rgb, pred_rgb], dim=2)

    save_image(combined, path)

def mario_gated_seg_loss(
    pred_seg: torch.Tensor,       # [B, C, H, W] logity
    target_seg: torch.Tensor,     # [B, C, H, W] one-hot
    pred_physics: torch.Tensor,   # [B, 6] — indexy 4,5 jsou cx,cy
    class_weights: torch.Tensor,
    mario_channel: int = 2,
    gate_sigma: float = 0.15,     # šířka gaussiánu kolem predikované pozice
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pro Mario kanál omezí kde smí predikce existovat pomocí gaussiánu
    kolem physics-predikované pozice. Ostatní kanály počítají loss normálně.
    """
    B, C, H, W = pred_seg.shape
    device = pred_seg.device

    # --- Gaussian prior kolem predikované pozice ---
    pred_cx = torch.clamp(pred_physics[:, 4], 0.0, 1.0)
    pred_cy = torch.clamp(pred_physics[:, 5], 0.0, 1.0)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    cx = pred_cx.view(B, 1, 1)
    cy = pred_cy.view(B, 1, 1)
    dist2 = (grid_x.unsqueeze(0) - cx)**2 + (grid_y.unsqueeze(0) - cy)**2
    # Gaussián — 1.0 u predikované pozice, blízko 0 daleko od ní
    gate = torch.exp(-dist2 / (2 * gate_sigma**2)).unsqueeze(1)  # [B, 1, H, W]

    # --- BCE loss ---
    raw_bce = F.binary_cross_entropy_with_logits(pred_seg, target_seg, reduction='none')

    # Pro Mario kanál vynásob BCE gaussiánem — loss daleko od predikce → ~0
    bce_gated = raw_bce.clone()
    bce_gated[:, mario_channel:mario_channel+1] *= gate

    loss_bce = (bce_gated * class_weights).mean()

    # --- Dice loss pouze v oblasti gatu ---
    pred_mario  = torch.sigmoid(pred_seg[:, mario_channel:mario_channel+1])
    target_mario = target_seg[:, mario_channel:mario_channel+1]

    # Omez predikci i target na oblast gatu
    pred_mario_gated   = pred_mario  * gate
    target_mario_gated = target_mario * gate

    eps = 1e-6
    intersection = (pred_mario_gated * target_mario_gated).flatten(1).sum(dim=1)
    union        = pred_mario_gated.flatten(1).sum(dim=1) + target_mario_gated.flatten(1).sum(dim=1)
    mario_dice   = 1.0 - ((2 * intersection + eps) / (union + eps)).mean()

    # Dice pro ostatní kanály bez gatu
    other_dice = dice_loss(
        torch.cat([pred_seg[:, :mario_channel], pred_seg[:, mario_channel+1:]], dim=1),
        torch.cat([target_seg[:, :mario_channel], target_seg[:, mario_channel+1:]], dim=1),
    )

    loss_dice = mario_dice + other_dice

    return loss_bce, loss_dice

def train_rollout(
    transformer,
    decoder,
    compressor,
    dataloader,
    optimizer,
    accelerator,
    rollout_steps=10,
    seq_len=4,
    num_epochs=20,
    detach=False,
    complex_loss=False
):
    transformer.train()
    decoder.train()
    compressor.train()

    bce = nn.BCEWithLogitsLoss(reduction='none')
    
    device = accelerator.device
    class_weights = torch.tensor([
        1.0,   # 0: background
        2.0,   # 1: ground
        20.0,  # 2: mario
        10.0,   # 3: block
        15.0,  # 4: goomba/enemy
        0.5,   # 5: cloud
        5.0,   # 6: pipe
        0.5,   # 7: bush
    ], device=device).view(1, 8, 1, 1)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            device = accelerator.device

            for key in ("all_segs", "all_cam_x", "all_cam_y"):
                batch[key] = batch[key].to(device)

            # --- Inicializace z GT snímků (pouze jednou na začátku) ---
            states = build_state_sequence(batch, seq_len, transformer, compressor, device)
            # states: [B, seq_len, total_state_dim]

            loss_accum = torch.tensor(0.0, device=device)

            for step in range(rollout_steps):
                t_target = seq_len + step  # index GT targetu v batchi

                actions = batch["all_actions"][:, step:step + seq_len].to(device)

                pred     = transformer(states, actions)
                pred_seg, pred_seg_aux = decoder(pred)

                # Mario kanál z predikované fyziky
                # pred_cx = pred["physics"][:, 4]
                # pred_cy = pred["physics"][:, 5]
                # pred_mario_logits = create_mario_map(torch.clamp(pred_cx, 0.0, 1.0), torch.clamp(pred_cy, 0.0, 1.0), size=128) * 20.0 - 10.0
                # pred_seg[:, 2:3] = pred_mario_logits

                # --- GT targety pro tento krok ---
                target_physics = build_target_state(batch, t_target).to(device)

                unwrapped = accelerator.unwrap_model(transformer)
                with torch.no_grad():
                    next_frame    = batch["all_frames"][:, t_target].to(device)
                    z_next        = unwrapped.encode_to_latent(next_frame)
                    target_visual = compressor(z_next)           # [B, visual_embed_dim]

                target_seg = build_segmentation_target(batch, t_target).to(device)
                
                target_seg_aux = F.interpolate(target_seg, size=(32, 32), mode='nearest')

                # --- Loss ---
                loss_physics = F.mse_loss(pred["physics"], target_physics)
                loss_visual  = F.mse_loss(pred["visual"],  target_visual)
                
                # raw_bce  = F.binary_cross_entropy_with_logits(pred_seg, target_seg, reduction='none')
                # loss_bce = (raw_bce * class_weights).mean()
                # loss_dice     = dice_loss(pred_seg, target_seg)
                # loss_boundary = boundary_loss(pred_seg, target_seg)

                raw_bce_aux = F.binary_cross_entropy_with_logits(pred_seg_aux, target_seg_aux, reduction='none')
                loss_seg_aux = (raw_bce_aux * class_weights).mean() + dice_loss(pred_seg_aux, target_seg_aux)

                if complex_loss:
                    loss_bce, loss_dice = mario_gated_seg_loss(
                        pred_seg, target_seg, pred["physics"], class_weights
                    )
                    loss_boundary = boundary_loss(pred_seg, target_seg)

                    loss_seg = loss_bce + 2.0 * loss_dice + 1.0 * loss_boundary + 0.5 * loss_seg_aux
                else:
                    raw_bce  = F.binary_cross_entropy_with_logits(pred_seg, target_seg, reduction='none')
                    loss_bce = (raw_bce * class_weights).mean()
                    loss_seg = loss_bce + 0.2 * loss_seg_aux

                # Pozdější kroky mají menší váhu — chyba se přirozeně akumuluje
                # step_weight = 0.9 ** step
                
                loss_latent_l2 = torch.mean(pred["latent_state"] ** 2)

                # step_loss = step_weight * (
                #     3.0 * loss_physics +
                #     2.0 * loss_visual  +
                #     12.0 * loss_seg +
                #     0.01 * loss_latent_l2 # Udržuje hodnoty blízko nule
                # )
                step_loss = (
                    3.0 * loss_physics +
                    2.0 * loss_visual  +
                    12.0 * loss_seg +
                    0.01 * loss_latent_l2 # Udržuje hodnoty blízko nule
                )

                loss_accum = loss_accum + step_loss / rollout_steps

                # --- Autoregresivní update stavu ---
                # Posun okna: zahoď nejstarší krok, přidej predikovaný
                # pred["full"] = [physics | visual | latent_state]
                if detach:
                    next_state = pred["full"].detach().unsqueeze(1)  # [B, 1, total_state_dim]
                else:
                    next_state = pred["full"].unsqueeze(1)  # [B, 1, total_state_dim]
                states = torch.cat([states[:, 1:], next_state], dim=1)

            accelerator.backward(loss_accum)
            accelerator.clip_grad_norm_(
                list(transformer.parameters())
                + list(decoder.parameters())
                + list(compressor.parameters()),
                1.0
            )
            optimizer.step()
            total_loss += loss_accum.item()

            if i % 50 == 0:
                frame = batch["all_frames"][:, seq_len + rollout_steps - 1]
                accelerator.wait_for_everyone() 
                if accelerator.is_main_process:
                    save_segmentation_debug(
                        pred_seg.detach().cpu(),
                        target_seg.detach().cpu(),
                        f"debug/debug_rollout{rollout_steps}_epoch{epoch}_iter{i}.png",
                        frame.detach().cpu()
                    )

        if accelerator.is_main_process:
            print(f"[Rollout ({rollout_steps} steps)] Epoch {epoch+1}: {total_loss/len(dataloader):.4f}")
            
        save_checkpoints(
            transformer=transformer,
            decoder=decoder,
            compressor=compressor,
            accelerator=accelerator,
            epoch=epoch,
            save_dir="./checkpoints_transformer",
            phase="rollout"
        )

def save_checkpoints(
    transformer, 
    decoder, 
    compressor, 
    accelerator, 
    epoch, 
    save_dir="checkpoints",
    phase="std"
):
    """
    Uloží váhy všech komponent world modelu bezpečně s ohledem na multi-GPU trénování.
    """
    # Ukládáme vždy pouze na hlavním procesu (rank 0)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        
        # Rozbalení modelů pro odstranění případného 'module.' prefixu z DDP
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_decoder = accelerator.unwrap_model(decoder)
        unwrapped_compressor = accelerator.unwrap_model(compressor)

        # Vytvoření slovníku se všemi váhami a metadaty
        checkpoint = {
            "epoch": epoch,
            "transformer_state_dict": unwrapped_transformer.state_dict(),
            "decoder_state_dict": unwrapped_decoder.state_dict(),
            "compressor_state_dict": unwrapped_compressor.state_dict(),
        }
        
        save_path = os.path.join(save_dir, f"checkpoint_{phase}.pt")
        
        # Samotné uložení
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint úspěšně uložen do: {save_path}")
        

def load_transformer_checkpoints(
    checkpoint_path,
    transformer,
    decoder,
    compressor,
    accelerator=None,
    device="cpu"
):
    """
    Načte váhy ze souboru do modelů.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nebyl nalezen na cestě: {checkpoint_path}")

    print(f"Načítám checkpoint z: {checkpoint_path}")
    
    # Načtení slovníku z disku. Nejbezpečnější je načíst ho nejdřív do RAM (device='cpu')
    # a samotné modely si ho pak přesunou na své aktuální GPU.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Bezpečné rozbalení modelů (stejně jako při ukládání), 
    # pro případ, že modely už prošly přes accelerator.prepare()
    t_model = accelerator.unwrap_model(transformer) if accelerator else transformer
    d_model = accelerator.unwrap_model(decoder) if accelerator else decoder
    c_model = accelerator.unwrap_model(compressor) if accelerator else compressor

    # Nahrání vah z dictionary do samotných modelů
    t_model.load_state_dict(checkpoint["transformer_state_dict"])
    d_model.load_state_dict(checkpoint["decoder_state_dict"])
    c_model.load_state_dict(checkpoint["compressor_state_dict"])

    print(f"✅ Váhy úspěšně načteny.")

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr
        
def train(transformer, decoder, compressor, dataloader, 
                     dataloader_long, optimizer, accelerator):
    
    # Fáze 1: Teacher forcing, žádná autoregrese
    train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=1, seq_len=4, num_epochs=80, detach=True)
    train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=1, seq_len=4, num_epochs=15, detach=True, complex_loss=True)
    
    # Fáze 2: Krátké rollouty s detach — stabilizace
    for steps in [2, 3, 5, 8, 12]:
        train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=steps, num_epochs=10, detach=True)
        train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=steps, num_epochs=3, detach=True, complex_loss=True)

    # Fáze 3: Střední rollouty
    set_lr(optimizer, 5e-5)
    for steps in [15, 20, 30]:
        train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=steps, num_epochs=20, detach=False)
        train_rollout(transformer, decoder, compressor, dataloader, optimizer, accelerator=accelerator, rollout_steps=steps, num_epochs=3, detach=False, complex_loss=True)
    
    # Fáze 4: Dlouhé rollouty pouze na dlouhém datasetu
    set_lr(optimizer, 1e-5)
    train_rollout(transformer, decoder, compressor, dataloader_long, optimizer, accelerator=accelerator, rollout_steps=50,  num_epochs=30, detach=False)
    train_rollout(transformer, decoder, compressor, dataloader_long, optimizer, accelerator=accelerator, rollout_steps=50,  num_epochs=3, detach=False, complex_loss=True)
    train_rollout(transformer, decoder, compressor, dataloader_long, optimizer, accelerator=accelerator, rollout_steps=100, num_epochs=30, detach=False)
    train_rollout(transformer, decoder, compressor, dataloader_long, optimizer, accelerator=accelerator, rollout_steps=100, num_epochs=3, detach=False, complex_loss=True)

if __name__ == "__main__":
    model = GameStateTransformer()
    decoder = SegmentationDecoder()
    compressor = LatentCompressor()
    
    # load_transformer_checkpoints(
    #     "checkpoints_transformer/checkpoint_rollout.pt",
    #     model,
    #     decoder,
    #     compressor,
    #     accelerator=None,
    #     device="cpu"
    # )
    
    data_loader = get_rollout_dataloader(
        root_dir="../data-generation/super-mario-bros/collected_data",
        seg_dir="mario_data_seg",
        batch_size=128,
        shuffle=True,
        num_workers=4,
        image_size=256,
        num_actions=4,
        num_frames=4,
        rollout_steps=30,
        stride=30,
        max_episodes=400
    )
    dataloader_long = get_rollout_dataloader(
        root_dir="../data-generation/super-mario-bros/collected_data",
        seg_dir="mario_data_seg",
        batch_size=16,
        shuffle=True,
        num_workers=4,
        image_size=256,
        num_actions=4,
        num_frames=4,
        rollout_steps=100,
        stride=100,
        max_episodes=400
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()) + list(compressor.parameters()),
        lr=1e-4
    )
    accelerator = Accelerator(mixed_precision="bf16")
    
    model, decoder, compressor, optimizer, data_loader, dataloader_long = accelerator.prepare(model, decoder, compressor, optimizer, data_loader, dataloader_long)
    
    train(model, decoder, compressor, dataloader=data_loader, dataloader_long=dataloader_long, optimizer=optimizer, accelerator=accelerator)
    