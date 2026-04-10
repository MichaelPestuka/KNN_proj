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
        
        # Latent state generujeme absolutně, protože Tanh() ho drží v bezpečných mezích [-1, 1]
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
        self.pre = nn.Linear(in_dim, 32 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64×64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),  # 128×128
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1),                       # [B, num_classes, 128, 128]
        )

    def forward(self, pred: dict):
        z = torch.cat([pred["visual"], pred["latent_state"]], dim=-1)
        x = self.pre(z).view(-1, 32, 8, 8)
        return self.net(x)
    
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
    detach=False
):
    transformer.train()
    decoder.train()
    compressor.train()

    bce = nn.BCEWithLogitsLoss(reduction='none')
    
    device = accelerator.device
    class_weights = torch.tensor([
        1.0,   # 0: background
        2.0,   # 1: ground
        15.0,  # 2: mario
        5.0,   # 3: block
        10.0,  # 4: goomba/enemy
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
                pred_seg = decoder(pred)

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

                # --- Loss ---
                loss_physics = F.mse_loss(pred["physics"], target_physics)
                loss_visual  = F.mse_loss(pred["visual"],  target_visual)
                
                raw_seg_loss = bce(pred_seg, target_seg)
                loss_seg = (raw_seg_loss * class_weights).mean()

                # Pozdější kroky mají menší váhu — chyba se přirozeně akumuluje
                step_weight = 0.9 ** step
                
                loss_latent_l2 = torch.mean(pred["latent_state"] ** 2)

                step_loss = step_weight * (
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
        
        # Volitelně můžeš přidat i stav optimizeru, pokud bys chtěl umět navázat v trénování:
        # "optimizer_state_dict": optimizer.state_dict()
        
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

if __name__ == "__main__":
    model = GameStateTransformer()
    decoder = SegmentationDecoder()
    compressor = LatentCompressor()
    load_transformer_checkpoints(
        "checkpoints_transformer/checkpoint_rollout.pt",
        model,
        decoder,
        compressor,
        accelerator=None,
        device="cpu"
    )
    
    data_loader = get_rollout_dataloader(
        root_dir="../data-generation/super-mario-bros/collected_data",
        seg_dir="mario_data_seg",
        batch_size=128,
        shuffle=True,
        num_workers=4,
        image_size=256,
        num_actions=4,
        num_frames=4,
        rollout_steps=10,
        stride=10,
        max_episodes=200
    )
    data_loader_bigger = get_rollout_dataloader(
        root_dir="../data-generation/super-mario-bros/collected_data",
        seg_dir="mario_data_seg",
        batch_size=64,
        shuffle=True,
        num_workers=4,
        image_size=256,
        num_actions=4,
        num_frames=4,
        rollout_steps=30,
        stride=30,
        max_episodes=200
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()) + list(compressor.parameters()),
        lr=1e-5
    )
    accelerator = Accelerator(mixed_precision="bf16")
    
    model, decoder, compressor, optimizer, data_loader, data_loader_bigger = accelerator.prepare(model, decoder, compressor, optimizer, data_loader, data_loader_bigger)
    
    # train_rollout(model, decoder, compressor, data_loader, optimizer, accelerator=accelerator, rollout_steps=1, seq_len=4, num_epochs=70)
    
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 5e-4
        
    train_rollout(model, decoder, compressor, data_loader, optimizer, accelerator=accelerator, rollout_steps=1, seq_len=4, num_epochs=40)
    
    train_rollout(model, decoder, compressor, data_loader, optimizer, accelerator=accelerator, rollout_steps=5, seq_len=4, num_epochs=20)
    
    train_rollout(model, decoder, compressor, data_loader, optimizer, accelerator=accelerator, rollout_steps=10, seq_len=4, num_epochs=20)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-5
    
    train_rollout(model, decoder, compressor, data_loader_bigger, optimizer, accelerator=accelerator, rollout_steps=20, seq_len=4, num_epochs=20)
    
    train_rollout(model, decoder, compressor, data_loader_bigger, optimizer, accelerator=accelerator, rollout_steps=30, seq_len=4, num_epochs=20)
