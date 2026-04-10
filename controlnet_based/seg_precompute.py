import json
import numpy as np
from PIL import Image
import os
import cv2
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from PIL import ImageDraw
from data_loader import mario_centroid_from_seg

COLOR_RANGES = {
    "sky":    [((0, 0, 6), (179, 255, 255))],
    "mario":  [((16, 200, 150), (20, 255, 190)),
               ((0, 220, 190), (10, 255, 215)),
               ((22, 200, 80), (34, 255, 140))],
    "cloud":  [((0, 0, 150), (120, 40, 255))],
    "bush": [((50, 220, 150), (70, 255, 190)),
             ((25, 210, 200), (40, 255, 255))]
}
CLASS_MAPPING = {
    "ground": 1,
    "block": 3,
    "goomba": 4,
    "pipe": 6,
}

def load_texture_templates(textures_dir: str):
    """
    Načte všechny textury z podsložek a přiřadí jim ID třídy.
    Tato funkce se volá jen JEDNOU před spuštěním smyčky.
    """
    templates = []
    
    for class_name in os.listdir(textures_dir):
        class_dir = os.path.join(textures_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        if class_name not in CLASS_MAPPING:
            print(f"⚠️ Ignoruji neznámou složku textur: {class_name}")
            continue
            
        class_id = CLASS_MAPPING[class_name]
        
        for tex_file in os.listdir(class_dir):
            if not tex_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            tex_path = os.path.join(class_dir, tex_file)
            # Načteme obrázek a převedeme z BGR (default OpenCV) do RGB
            tex_img = cv2.imread(tex_path)
            if tex_img is not None:
                tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
                templates.append((class_id, tex_img))
                
    print(f"✅ Úspěšně načteno {len(templates)} texturových šablon.")
    return templates

def combined_segment_frame(img_rgb: np.ndarray, templates: list, threshold: float = 0.85) -> np.ndarray:
    """
    Vytvoří segmentační mapu kombinací barev a textur podle Z-indexu (priorit vrstev).
    """
    H, W = img_rgb.shape[:2]
    
    # KROK 1: Výchozí vrstva = Pozadí (Třída 0)
    seg_map = np.zeros((H, W), dtype=np.uint8)
    
    # Příprava HSV pro color matching
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    def apply_color(ranges, class_id):
        mask = np.zeros((H, W), dtype=bool)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi)).astype(bool)
        seg_map[mask] = class_id

    # KROK 2: Aplikace Mraků (Color) - jsou úplně vzadu
    apply_color(COLOR_RANGES["cloud"], 5)
    apply_color(COLOR_RANGES["bush"], 7)
    
    # KROK 3: Aplikace Textur (Block, Ground, Pipe, Goomba)
    matches = []
    for class_id, template in templates:
        th, tw, _ = template.shape
        if th > H or tw > W:
            continue
            
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
        
        # Specifický práh pro Goombu, jinak default
        if class_id == 4:
            loc = np.where(res >= 0.5)
        else:
            loc = np.where(res >= threshold)
            
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            matches.append((score, class_id, pt[0], pt[1], tw, th))
            
    # Vyřešení překryvů mezi texturami navzájem
    matches.sort(key=lambda x: x[0], reverse=True)
    filled_textures = np.zeros((H, W), dtype=bool)
    
    for score, class_id, x, y, w, h in matches:
        roi_filled = filled_textures[y:y+h, x:x+w]
        mask = ~roi_filled
        
        # Zápis do výsledné mapy (textury přepíší mraky i pozadí)
        seg_map[y:y+h, x:x+w][mask] = class_id
        filled_textures[y:y+h, x:x+w] = True
        
    # KROK 4: Aplikace Maria (Color) - má nejvyšší prioritu, překreslí vše pod ním
    apply_color(COLOR_RANGES["mario"], 2)
    
    return seg_map

def process_single_episode(episode_name, root_dir, output_dir, templates, image_size):
    ep_path = os.path.join(root_dir, episode_name)
    if not os.path.isdir(ep_path):
        return None

    json_file = next((os.path.join(ep_path, f) for f in os.listdir(ep_path) if f.endswith(".json")), None)
    if not json_file:
        return None

    with open(json_file) as f:
        data = json.load(f)

    out_ep = os.path.join(output_dir, episode_name)
    os.makedirs(out_ep, exist_ok=True)

    for frame_meta in data["frames"]:
        img_path = os.path.join(ep_path, frame_meta["filename"])
        
        # Segmentace na nativním rozlišení
        img_original = np.array(Image.open(img_path).convert("RGB"))
        seg_original = combined_segment_frame(img_original, templates, threshold=0.7)

        # Resize segmentační mapy přes NEAREST
        seg_resized = np.array(
            Image.fromarray(seg_original).resize((image_size, image_size), Image.NEAREST)
        )

        mask_name = frame_meta["filename"].replace(".jpg", ".npy")
        np.save(os.path.join(out_ep, mask_name), seg_resized)

    return f"✅ {episode_name}: Zpracováno {len(data['frames'])} masek."


# 2. Tvoje upravená hlavní funkce
def precompute_seg_masks(
    root_dir: str, 
    output_dir: str, 
    textures_dir: str, 
    image_size: int = 256,
    max_episodes: int = None,
    num_workers: int = 24
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ Načítám templates (provádí se jen jednou)...")
    templates = load_texture_templates(textures_dir)

    # 1. Načteme a seřadíme složky
    all_episodes = sorted(os.listdir(root_dir))
    
    # 2. Aplikujeme stejné oříznutí
    if max_episodes is not None:
        all_episodes = all_episodes[:max_episodes]
        print(f"🚀 Omezeno: Bude precomputováno max {max_episodes} epizod.")

    # Pokud neurčíme počet jader, využijeme všechna dostupná CPU minus 1 (ať úplně nezamrzneme PC)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    print(f"🔥 Spouštím multiprocessing na {num_workers} jádrech...")

    # Zabalíme parametry, které se v průběhu nemění, do jedné fixní funkce pomocí partial
    worker_fn = partial(
        process_single_episode, 
        root_dir=root_dir, 
        output_dir=output_dir, 
        templates=templates, 
        image_size=image_size
    )

    # 3. Samotný multiprocesing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Nasázíme úkoly do executoru
        futures = {executor.submit(worker_fn, ep): ep for ep in all_episodes}
        
        # as_completed nám vrací výsledky tak rychle, jakmile jsou hotové
        for future in as_completed(futures):
            episode = futures[future]
            try:
                result = future.result()
                if result: # Vypíše hlášku z workeru
                    print(result)
            except Exception as exc:
                print(f"❌ Epizoda {episode} vygenerovala výjimku: {exc}")
                
    print("🎉 Všechny epizody byly úspěšně precomputovány!")
        
        
def test_segmentation_on_single_frame(image_path: str, textures_dir: str, output_path: str, threshold: float = 0.75, upscale_factor: int = 4):
    if not os.path.exists(image_path):
        print(f"❌ Obrázek neexistuje: {image_path}")
        return

    print("Načítám šablony textur...")
    templates = load_texture_templates(textures_dir)
    
    if not templates:
        print("❌ Žádné textury nebyly nalezeny!")
        return

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    print("Segmentuji obrázek kombinovanou metodou...")
    seg_map = combined_segment_frame(img_np, templates, threshold=threshold)
    
    cx, cy = mario_centroid_from_seg(seg_map)
    if cx is not None:
        print(f"Mario centroid: cx={cx:.3f}, cy={cy:.3f}")
    else:
        print("⚠️ Mario centroid nenalezen.")

    palette = np.array([
        [25,  25,  25 ],
        [180, 100, 20 ],
        [220, 50,  50 ],
        [255, 150, 50 ],
        [140, 80,  180],
        [100, 200, 255],
        [50,  180, 50 ],
        [75,  255, 100 ],
    ], dtype=np.uint8)

    seg_map = np.clip(seg_map, 0, len(palette) - 1)
    seg_rgb = palette[seg_map]

    out_img = Image.fromarray(seg_rgb)
    
    if upscale_factor > 1:
        w, h = out_img.size
        new_size = (w * upscale_factor, h * upscale_factor)
        out_img = out_img.resize(new_size, Image.NEAREST)
        img     = img.resize(new_size, Image.NEAREST)

    # Vykreslení čtverce na obou obrázcích
    if cx is not None:
        box_half_w = int(img.width  * 0.03)
        box_half_h = int(img.height * 0.06)

        cx_px = int(cx * img.width)
        cy_px = int(cy * img.height)
        box   = [cx_px - box_half_w, cy_px - box_half_h,
                 cx_px + box_half_w, cy_px + box_half_h]

        for canvas in (img, out_img):
            draw = ImageDraw.Draw(canvas)
            draw.rectangle(box, outline=(255, 255, 0), width=2)

    total_width = img.width + out_img.width
    side_by_side = Image.new('RGB', (total_width, img.height))
    side_by_side.paste(img,     (0, 0))
    side_by_side.paste(out_img, (img.width, 0))

    side_by_side.save(output_path)
    print(f"✅ Uloženo do: {output_path}")

# for i in range(150):
#     TEST_IMAGE = f"../mario_data/combined_w1s1_4b27fffb/frame_4b27fffb_000{i*3 + 230}.jpg"
#     TEXTURES = "./textures"
#     OUTPUT = f"debug/debug_segmentation_output{i}.png"

#     test_segmentation_on_single_frame(
#         image_path=TEST_IMAGE,
#         textures_dir=TEXTURES,
#         output_path=OUTPUT,
#         threshold=0.7,    # Pokud algoritmus něco nenajde, zkus snížit např. na 0.75
#         upscale_factor=4   # Uloží to 4x větší pro lepší zkoumání
#     )

precompute_seg_masks("../data-generation/super-mario-bros/collected_data", "mario_data_seg", "textures", max_episodes=200)