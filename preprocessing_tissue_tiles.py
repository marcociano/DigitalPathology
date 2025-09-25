import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image

# ---------- util base ----------

def suppress_specular(img_bgr: np.ndarray) -> np.ndarray:
    """Rimuove riflessi: (S basso & V alto) + (L alto & a*,b* ~ 0) → inpainting."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    mask1 = ((S < 30) & (V > 230))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mask2 = (L > 235) & (np.abs(A - 128) < 6) & (np.abs(B - 128) < 6)

    mask = ((mask1 | mask2).astype(np.uint8)) * 255
    if mask.mean() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

def clahe_L(img_bgr: np.ndarray, clip=2.0, tiles=(8,8)) -> np.ndarray:
    """Uniforma il contrasto in LAB solo sul canale L* (stain-invariant)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    L = clahe.apply(L)
    lab = cv2.merge([L, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def unsharp(img: np.ndarray, ksize: int = 5, amount: float = 0.4) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

# ---------- segmentazione tessuto ----------

def tissue_mask_bgr(
    img_bgr: np.ndarray,
    sat_thresh: int = 20, v_min: int = 25, v_max: int = 245,
    morph_open: int = 3, morph_close: int = 5, min_area: int = 400
) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    base = (S > sat_thresh) & (V > v_min) & (V < v_max)
    mask = base.astype(np.uint8) * 255

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

# ---------- bbox & allineamento ----------

def largest_bbox(mask: np.ndarray, margin: float = 0.04) -> Tuple[int, int, int, int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    x, y, w, h = (stats[k, cv2.CC_STAT_LEFT],
                  stats[k, cv2.CC_STAT_TOP],
                  stats[k, cv2.CC_STAT_WIDTH],
                  stats[k, cv2.CC_STAT_HEIGHT])
    H, W = mask.shape[:2]
    mx = int(margin * W); my = int(margin * H)
    x = max(0, x - mx); y = max(0, y - my)
    w = min(W - x, w + 2 * mx); h = min(H - y, h + 2 * my)
    return int(x), int(y), int(w), int(h)

def principal_axis_angle(mask: np.ndarray) -> float:
    """Stima l'angolo (radianti) dell'asse principale del tessuto via PCA sui pixel = 255."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0
    X = np.vstack([xs, ys]).T.astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X)-1, 1)
    w, v = np.linalg.eigh(cov)
    pc = v[:, np.argmax(w)]
    angle = np.arctan2(pc[1], pc[0])  # in radianti
    return float(angle)

def rotate_keep_size(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def crop_to_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def pad_to_square(img: np.ndarray, border_value=(255, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)

def resize(img: np.ndarray, size: int = 512) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

# ---------- pipeline principale ----------

def prepare_image_for_snn(
    path: str,
    out_size: int = 512,
    return_debug: bool = False
) -> Tuple[Image.Image, Optional[Dict[str, Any]]]:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Immagine non trovata: {path}")

    # 1) riflessi + CLAHE(L*) + leggera nitidezza
    img_bgr = suppress_specular(img_bgr)
    img_bgr = clahe_L(img_bgr, clip=2.0, tiles=(8,8))
    img_bgr = unsharp(img_bgr, 5, 0.4)

    # 2) mask + allineamento per asse principale
    mask = tissue_mask_bgr(img_bgr)
    theta = principal_axis_angle(mask)
    # ruota per allineare l'asse verticale (in gradi, verso orario negativo)
    img_bgr = rotate_keep_size(img_bgr, -np.degrees(theta))
    mask = rotate_keep_size(mask, -np.degrees(theta))

    # 3) crop + padding + resize
    x, y, w, h = largest_bbox(mask, margin=0.05)
    crop_bgr = crop_to_bbox(img_bgr, (x, y, w, h))
    crop_bgr = pad_to_square(crop_bgr, (255, 255, 255))
    crop_bgr = resize(crop_bgr, out_size)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)

    debug = None
    if return_debug:
        dbg = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        debug = {"bbox": (x, y, w, h), "mask": mask, "angle_deg": -np.degrees(theta), "preview_rgb": dbg}
    return pil, debug
