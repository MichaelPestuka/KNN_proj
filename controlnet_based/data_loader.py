import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from sklearn.cluster import DBSCAN

# --- CONFIG ---
ACTION_LIST = ["left", "right", "A", "B", "NOOP"]
ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTION_LIST)}


def encode_action(action_list):
    """
    Convert list of actions into multi-hot vector.
    """
    vec = torch.zeros(len(ACTION_LIST), dtype=torch.float32)
    if action_list is None:
        vec[ACTION_TO_INDEX["NOOP"]] = 1.0
        return vec
    for a in action_list:
        if a in ACTION_TO_INDEX:
            vec[ACTION_TO_INDEX[a]] = 1.0
    return vec

class MarioDataset(Dataset):
    def __init__(
        self,
        root_dir="mario_data",
        num_frames=4,
        num_actions=12,
        image_size=256,
        max_samples_per_episode=None,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.image_size = image_size
        self.max_samples_per_episode = max_samples_per_episode

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # [0,1]
        ])

        self.samples = []
        self._build_index()

    def _build_index(self):
        """
        Build index of all valid samples across all episodes.
        """
        for episode_name in os.listdir(self.root_dir):
            episode_path = os.path.join(self.root_dir, episode_name)

            if not os.path.isdir(episode_path):
                continue

            # find json file
            json_file = None
            for f in os.listdir(episode_path):
                if f.endswith(".json"):
                    json_file = os.path.join(episode_path, f)
                    break

            if json_file is None:
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            frames = data["frames"]
            frames = sorted(frames, key=lambda x: x["frame"])
                            
            total = len(frames)

            # sliding window
            samples_added = 0
            for t in range(total):
                if t + 1 >= total:
                    continue

                # skip if next frame is terminal
                if frames[t]["done"]:
                    continue

                self.samples.append({
                    "episode_path": episode_path,
                    "frames": frames,
                    "t": t
                })
                
                samples_added += 1
                
                # Pokud jsme dosáhli maximálního počtu vzorků pro tuto epizodu, 
                # ukončíme cyklus a jdeme na další epizodu.
                if self.max_samples_per_episode is not None and samples_added >= self.max_samples_per_episode:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        episode_path = sample["episode_path"]
        frames_meta = sample["frames"]
        t = sample["t"]

        # --- LOAD FRAME SEQUENCE ---
        frame_tensors = []
        for i in range(t - self.num_frames + 1, t + 1):
            if i < 0:
                # PADDING pro začátek epizody
                black_frame = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                frame_tensors.append(black_frame)
            else:
                fname = frames_meta[i]["filename"]
                path = os.path.join(episode_path, fname)

                img = Image.open(path).convert("RGB")
                img = self.transform(img)
                frame_tensors.append(img)

        frames_tensor = torch.stack(frame_tensors)  # (num_frames, C, H, W)

        # --- LOAD ACTION HISTORY ---
        action_tensors = []
        for i in range(t - self.num_actions + 1, t + 1):
            if i < 0:
                # PADDING pro chybějící akce
                encoded = encode_action(None)
                action_tensors.append(encoded)
            else:
                action = frames_meta[i]["action"]
                encoded = encode_action(action)
                action_tensors.append(encoded)

        actions_tensor = torch.stack(action_tensors)  # (num_actions, action_dim)

        # --- TARGET FRAME ---
        next_frame_meta = frames_meta[t + 1]
        next_path = os.path.join(episode_path, next_frame_meta["filename"])

        target_img = Image.open(next_path).convert("RGB")
        target_img = self.transform(target_img)
        
        next_info = next_frame_meta["info"]

        # --- TARGET COORDS ---
        raw_x = next_info["x_pos"]
        raw_y = next_info["y_pos"]

        # normalizace
        x_norm = raw_x / 3500.0
        y_norm = raw_y / 240.0

        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32)
        
        # --- CURRENT COORDS (t) ---
        curr_info = frames_meta[t]["info"]

        curr_x = curr_info["x_pos"] / 3400.0
        curr_y = min(max(curr_info["y_pos"], 0), 240) / 240
        
        if t > 0:
            prev_info = frames_meta[t-1]["info"]
            prev_x = prev_info["x_pos"] / 3400.0
            prev_y = min(max(prev_info["y_pos"], 0), 240) / 240
        else:
            prev_x, prev_y = curr_x, curr_y
        
        vx = (curr_x - prev_x) * 10.0
        vy = (curr_y - prev_y) * 10.0

        vx_tensor = torch.tensor(vx, dtype=torch.float32)
        vy_tensor = torch.tensor(vy, dtype=torch.float32)

        return {
            "frames": frames_tensor,          # (F, C, H, W)
            "actions": actions_tensor,        # (A, action_dim)
            "target": target_img,             # (C, H, W)
            "x": x_tensor,
            "y": y_tensor,
            "vx": vx_tensor,
            "vy": vy_tensor
        }


def get_dataloader(
    root_dir="mario_data",
    batch_size=8,
    shuffle=True,
    num_workers=4,
    num_frames=4,
    num_actions=40,
    image_size=256,
    max_samples_per_episode=None,
):
    dataset = MarioDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        num_actions=num_actions,
        image_size=image_size,
        max_samples_per_episode=max_samples_per_episode,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    
class MarioRolloutDataset(Dataset):
    def __init__(
        self,
        root_dir="mario_data",
        seg_dir="mario_data_seg",
        num_frames=4,
        num_actions=40,
        rollout_steps=15,
        stride=15, # 🚀 NOVÉ: O kolik snímků se okno posune. Výchozí hodnota bez překryvu.
        image_size=256,
        max_samples_per_episode=None,
        max_episodes=None,
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.rollout_steps = rollout_steps
        self.stride = stride
        self.image_size = image_size
        self.max_samples_per_episode = max_samples_per_episode
        self.seg_dir = seg_dir
        self.max_episodes = max_episodes

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # [0,1]
        ])

        self.samples = []
        self._build_index()

    def _build_index(self):
        # Nejdřív si seřadíme všechny dostupné epizody
        all_episodes = sorted(os.listdir(self.seg_dir))
        
        if self.max_episodes is not None:
            # Tip: Kdybys chtěl načítat POKAŽDÉ JINÉ náhodné epizody, odkomentuj tento řádek:
            # random.shuffle(all_episodes) 
            all_episodes = all_episodes[:self.max_episodes]
            print(f"Dataset omezen na max {self.max_episodes} epizod.")

        for episode_name in all_episodes:
            episode_path = os.path.join(self.root_dir, episode_name)
            if not os.path.isdir(episode_path): continue

            json_file = next((os.path.join(episode_path, f) for f in os.listdir(episode_path) if f.endswith(".json")), None)
            if not json_file: continue

            with open(json_file, "r") as f:
                data = json.load(f)

            frames = sorted(data["frames"], key=lambda x: x["frame"])
            total = len(frames)

            valid_starts = []
            
            for t in range(0, total, self.stride):
                if t + self.rollout_steps >= total:
                    continue
                
                invalid = False
                for step_idx in range(t, t + self.rollout_steps):
                    if frames[step_idx]["done"]:
                        invalid = True
                        break
                if invalid: continue

                valid_starts.append(t)
                
            if self.max_samples_per_episode and len(valid_starts) > self.max_samples_per_episode:
                valid_starts = random.sample(valid_starts, self.max_samples_per_episode)

            for t in valid_starts:
                self.samples.append({
                    "episode_path": episode_path,
                    "frames": frames,
                    "t": t
                })
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        episode_path = sample["episode_path"]
        frames_meta  = sample["frames"]
        t            = sample["t"]
        episode_name = os.path.basename(episode_path)

        frame_tensors = []
        seg_tensors   = []
        cam_x_coords  = []
        cam_y_coords  = []

        last_valid_cx, last_valid_cy = 0.5, 0.5  # fallback

        for i in range(t - self.num_frames + 1, t + self.rollout_steps + 1):
            if i < 0:
                frame_tensors.append(torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32))
                seg_tensors.append(torch.zeros((self.image_size, self.image_size), dtype=torch.long))
                cam_x_coords.append(last_valid_cx)
                cam_y_coords.append(last_valid_cy)
                continue

            fname = frames_meta[i]["filename"]

            # snímek
            img = self.transform(
                Image.open(os.path.join(episode_path, fname)).convert("RGB")
            )
            frame_tensors.append(img)

            # segmentační maska
            mask_path = os.path.join(
                self.seg_dir, episode_name, fname.replace(".jpg", ".npy")
            )
            seg_np = np.load(mask_path)  # uint8 [H, W]
            seg_tensors.append(torch.from_numpy(seg_np).long())

            # camera-space souřadnice Maria z centroidu class-2 pixelů
            cx, cy = mario_centroid_from_seg(seg_np)
            if cx is None:
                cx, cy = last_valid_cx, last_valid_cy
            else:
                last_valid_cx, last_valid_cy = cx, cy
            cam_x_coords.append(cx)
            cam_y_coords.append(cy)

        # akce
        action_tensors = []
        for i in range(t - self.num_actions + 1, t + self.rollout_steps + 1):
            if i < 0:
                action_tensors.append(encode_action(None))
            else:
                action_tensors.append(encode_action(frames_meta[i]["action"]))

        # physics (world coords — velocity, světová x pro scrolling logiku)
        VIEWPORT_H = 240.0
        WORLD_W    = 3400.0
        VEL_SCALE  = 10.0
        seq_start  = t - self.num_frames + 1

        x_coords, y_coords, vx_coords, vy_coords = [], [], [], []
        for i in range(seq_start, t + self.rollout_steps + 1):
            if i < 0:
                x_coords.append(0.0); y_coords.append(0.0)
                vx_coords.append(0.0); vy_coords.append(0.0)
                continue

            curr = frames_meta[i]["info"]
            x = curr["x_pos"] / WORLD_W
            y = min(max(curr["y_pos"], 0), VIEWPORT_H) / VIEWPORT_H

            if i == 0 or i == seq_start:
                vx, vy = 0.0, 0.0
            else:
                prev = frames_meta[i - 1]["info"]
                px   = prev["x_pos"] / WORLD_W
                py   = min(max(prev["y_pos"], 0), VIEWPORT_H) / VIEWPORT_H
                vx   = (x - px) * VEL_SCALE
                vy   = (y - py) * VEL_SCALE

            x_coords.append(x);   y_coords.append(y)
            vx_coords.append(vx); vy_coords.append(vy)

        return {
            "all_frames":   torch.stack(frame_tensors),
            "all_segs":     torch.stack(seg_tensors),          # [T, H, W]
            "all_actions":  torch.stack(action_tensors),
            "all_x":        torch.tensor(x_coords,    dtype=torch.float32),
            "all_y":        torch.tensor(y_coords,    dtype=torch.float32),
            "all_vx":       torch.tensor(vx_coords,   dtype=torch.float32),
            "all_vy":       torch.tensor(vy_coords,   dtype=torch.float32),
            "all_cam_x":    torch.tensor(cam_x_coords, dtype=torch.float32),  # [T] kamera
            "all_cam_y":    torch.tensor(cam_y_coords, dtype=torch.float32),  # [T] kamera
        }
        
def mario_centroid_from_seg(
    seg_np: np.ndarray,
    min_cluster_pixels: int = 3,
    eps: float = 8.0,
) -> tuple[float | None, float | None]:
    """
    Najde centroid Maria pomocí DBSCAN clusteringu + prostorového prioru.
    
    Mario je vždy přibližně v horizontálním středu obrazovky (x ∈ 0.3–0.7)
    a nikdy není v úplném vrcholu nebo spodku (y ∈ 0.1–0.95).
    """
    H, W = seg_np.shape
    ys, xs = np.where(seg_np == 2)

    if len(xs) < min_cluster_pixels:
        return None, None

    points = np.column_stack([xs, ys]).astype(np.float32)

    # DBSCAN — najde shluky, ignoruje osamělé falešné pixely jako noise (-1)
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(points)

    best_cluster = None
    best_score   = -1.0

    for label in set(labels):
        if label == -1:  # noise
            continue

        mask   = labels == label
        cx_px  = points[mask, 0].mean()
        cy_px  = points[mask, 1].mean()
        size   = mask.sum()

        # Skóre = velikost clusteru (největší validní cluster = Mario)
        if size > best_score:
            best_score   = size
            best_cluster = (cx_px, cy_px)

    if best_cluster is None:
        return None, None

    cx = best_cluster[0] / W
    cy = best_cluster[1] / H
    return cx, cy


def get_rollout_dataloader(
    root_dir="mario_data",
    seg_dir="mario_data_seg",
    batch_size=8,
    shuffle=True,
    num_workers=4,
    num_frames=4,
    num_actions=40,
    image_size=256,
    max_samples_per_episode=None,
    rollout_steps=15,
    stride=15,
    max_episodes=None
):
    dataset = MarioRolloutDataset(
        root_dir=root_dir,
        seg_dir=seg_dir,
        num_frames=num_frames,
        num_actions=num_actions,
        image_size=image_size,
        max_samples_per_episode=max_samples_per_episode,
        rollout_steps=rollout_steps,
        stride=stride,
        max_episodes=max_episodes
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )