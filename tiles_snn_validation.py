# tiles_snn_validation.py — Tiled SNN deterministica (multiscale, kNN-ratio simmetrico, RANSAC)
from typing import List, Tuple, Dict, Any
import os, random, hashlib
import numpy as np
import cv2

# --- determinismo forte (mettilo PRIMA di import torch/torchvision) ---
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # determinismo CUDA (se usi GPU)
random.seed(0)
np.random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

from preprocessing_tissue_tiles import prepare_image_for_snn, tissue_mask_bgr

def set_strict_determinism():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

# ======= PATH =======
SLIDE_PATH = "/Users/marcociano/Desktop/Tirocinio/DigitalPathology_Rendering/Images_for_SIFT/vetrine2_rit.jpg"
BLOCK_PATH = "/Users/marcociano/Desktop/Tirocinio/DigitalPathology_Rendering/Images_for_SIFT/tissue_block2.jpg"

# ======= PARAM =======
INPUT_SIZE = 512

FORCE_CPU = True

# multiscale tiling sul crop 512×512
TILE_SCALES = [160, 128, 96]
STRIDE_FACTOR = 0.4
TILE_MIN_TISSUE = 0.20

# orientazioni per invarianza
ORIENTATIONS = [0, 90, 180, 270]
USE_FLIP = True

# soglie distanza (cosine distance)
ABS_COS_THR_MIN = 0.50
ABS_COS_THR_MED = 0.57

# matching: kNN + ratio test (con ordinamento STABILE)
KNN_K = 2
RATIO_MAX = 0.90

# RANSAC deterministico su coordinate normalizzate [0,1]
RANSAC_ITERS = 2000
RANSAC_TOL_FRAC = 0.16
RANSAC_SEED = 42  # fisso

# paletti minimi (adattivi, con floor)
MIN_MATCHES_FLOOR = 6
MIN_TILE_COVERAGE = 0.05
MIN_INLIERS_FLOOR = 3
MIN_INLIER_RATIO = 0.25

# EPS per confronti al pelo
EPS = 1e-9

# ======= MODEL =======
class EmbeddingNet(nn.Module):
    def __init__(self, dim_out: int = 256):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4, backbone.avgpool
        )
        self.proj = nn.Linear(512, dim_out)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.tf = weights.transforms()

    @torch.inference_mode()
    def embed_pils(self, pil_list: List[Image.Image]) -> torch.Tensor:
        if not pil_list:
            return torch.empty(0, 256)
        dev = next(self.parameters()).device
        xs = [self.tf(p).unsqueeze(0) for p in pil_list]
        X = torch.cat(xs, dim=0).to(dev)
        feat = self.features(X).flatten(1)
        emb = F.normalize(self.proj(feat), p=2, dim=1)
        return emb

# ======= UTILS =======
def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def tile_coords(size: int, tile: int, stride: int) -> List[Tuple[int,int,int,int]]:
    coords = []
    for y in range(0, size - tile + 1, stride):
        for x in range(0, size - tile + 1, stride):
            coords.append((x, y, tile, tile))
    return coords

def extract_tissue_tiles_multiscale(pil_img: Image.Image,
                                    tile_scales: List[int],
                                    stride_factor: float,
                                    min_tissue: float) -> Tuple[List[Image.Image], List[Tuple[int,int,int,int]]]:
    bgr = pil_to_bgr(pil_img)
    mask = tissue_mask_bgr(bgr)  # 0/255
    mb = (mask > 0).astype(np.uint8)

    seen: set = set()
    tiles: List[Image.Image] = []
    boxes: List[Tuple[int,int,int,int]] = []

    H, W = bgr.shape[:2]
    for tile in tile_scales:
        stride = min(max(1, int(round(tile * stride_factor))), tile)
        for (x, y, w, h) in tile_coords(H, tile, stride):
            key = (x, y, tile, tile)
            if key in seen:
                continue
            m = mb[y:y+tile, x:x+tile]
            if m.size == 0 or m.mean() < float(min_tissue):
                continue
            crop = bgr[y:y+tile, x:x+tile]
            tiles.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            boxes.append((x, y, tile, tile))
            seen.add(key)
    return tiles, boxes

def augment_orientations(tiles: List[Image.Image]) -> List[List[Image.Image]]:
    all_views = []
    for t in tiles:
        views = [transforms.functional.rotate(t, ang) for ang in ORIENTATIONS]
        if USE_FLIP:
            views += [transforms.functional.hflip(v) for v in views]
        all_views.append(views)
    return all_views

@torch.inference_mode()
def embed_oriented_tiles(model: EmbeddingNet, tiles_views: List[List[Image.Image]]) -> torch.Tensor:
    flat = [v for views in tiles_views for v in views]
    return model.embed_pils(flat)

def oriented_min_dist_matrix(Es: torch.Tensor, Eb: torch.Tensor,
                             ns: int, nb: int, k: int) -> np.ndarray:
    # sim tutte-vs-tutte deterministica (stesse matrici → stessi risultati)
    sim = (Es @ Eb.T).cpu().numpy()  # (ns*k, nb*k)
    D = np.zeros((ns, nb), dtype=np.float32)
    for i in range(ns):
        i0, i1 = i*k, (i+1)*k
        block_i = sim[i0:i1, :]
        for j in range(nb):
            j0, j1 = j*k, (j+1)*k
            max_sim = np.max(block_i[:, j0:j1])
            D[i, j] = float(max(0.0, 1.0 - max_sim))
    return D

def knn_ratio_matches(D: np.ndarray, k: int = KNN_K, ratio_max: float = RATIO_MAX,
                      abs_min: float = ABS_COS_THR_MIN) -> List[Tuple[int,int,float]]:
    """
    Deterministico: ordina con np.lexsort (distanza, indice) per tie-break stabile.
    """
    Na, Nb = D.shape
    pairs = []
    for i in range(Na):
        order = np.lexsort((np.arange(Nb), D[i]))  # stable: prima per dist, poi per j
        if len(order) < 1:
            continue
        j1 = order[0]; d1 = D[i, j1]
        j2 = order[1] if len(order) > 1 else j1
        d2 = D[i, j2]
        if (d1 <= abs_min + EPS) and ((d1 + EPS) / (d2 + EPS) <= ratio_max + EPS):
            pairs.append((i, j1, float(d1)))
    # unicità lato block: tieni, per ogni j, il match con distanza minore (stabile)
    best_for_j: Dict[int, Tuple[int,int,float]] = {}
    for i, j, d1 in pairs:
        if (j not in best_for_j) or (d1 < best_for_j[j][2] - EPS) or \
           (abs(d1 - best_for_j[j][2]) <= EPS and i < best_for_j[j][0]):
            best_for_j[j] = (i, j, d1)
    out = [(v[0], v[1], v[2]) for v in best_for_j.values()]
    out.sort(key=lambda x: (x[2], x[0], x[1]))  # stabile
    return out

def symmetric_consistency(matches_ab: list[tuple[int,int,float]],
                          matches_ba: list[tuple[int,int,float]]) -> tuple[float,float,float]:
    """
    Restituisce (prec_ab, prec_ba, jaccard):
      - prec_ab = % di match A->B confermati anche da B->A
      - prec_ba = % di match B->A confermati anche da A->B
      - jaccard = Jaccard classico (facoltativo)
    """
    set_ab = {(i, j) for i, j, _ in matches_ab}
    set_ba_inv = {(j, i) for i, j, _ in matches_ba}

    inter = set_ab & set_ba_inv
    prec_ab = len(inter) / max(len(set_ab), 1)
    prec_ba = len(inter) / max(len(set_ba_inv), 1)
    jaccard = len(inter) / max(len(set_ab | set_ba_inv), 1)
    return prec_ab, prec_ba, jaccard


def quadrant_dispersion(boxes: List[Tuple[int,int,int,int]], idxs: List[int], size: int = 512) -> int:
    quads = set()
    for i in idxs:
        x, y, w, h = boxes[i]
        cx = x + w*0.5; cy = y + h*0.5
        qx = 0 if cx < size/2 else 1
        qy = 0 if cy < size/2 else 1
        quads.add((qx, qy))
    return len(quads)

def upper_tri(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]

def dist_stats(vec: np.ndarray) -> Dict[str, float]:
    v = vec.reshape(-1)
    if v.size == 0:
        return {"min":0,"p25":0,"med":0,"p75":0,"max":0}
    return {
        "min": float(np.min(v)),
        "p25": float(np.percentile(v, 25)),
        "med": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "max": float(np.max(v)),
    }

@torch.inference_mode()
def self_cosine_stats(E: torch.Tensor) -> Dict[str, float]:
    if E.shape[0] < 2:
        return {"min":0,"p25":0,"med":0,"p75":0,"max":0}
    sim = (E @ E.T).cpu().numpy()
    cos = 1.0 - sim
    return dist_stats(upper_tri(cos))

def ransac_similarity_norm(pts1, pts2, iters=RANSAC_ITERS, tol_frac=RANSAC_TOL_FRAC, seed=RANSAC_SEED):
    # RANSAC deterministico: RNG fisso + scelta campioni ordinata in caso di parità
    rng = np.random.default_rng(seed)
    pts1 = np.asarray(pts1, np.float32)
    pts2 = np.asarray(pts2, np.float32)
    N = len(pts1)
    if N < 2:
        return np.zeros((N,), bool), None
    scale = np.array([INPUT_SIZE, INPUT_SIZE], dtype=np.float32)
    A = pts1 / scale; B = pts2 / scale
    best_inl = np.zeros((N,), bool)
    best_sum = -1
    best_med = None
    for _ in range(iters):
        idx = rng.choice(N, 2, replace=False)
        idx = np.sort(idx)  # ordinamento deterministico dei campioni
        a0, a1 = A[idx]; b0, b1 = B[idx]
        v1, v2 = a1 - a0, b1 - b0
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        s = n2 / n1
        ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        t = b0 - s * (R @ a0)
        pred = (s * (R @ A.T)).T + t
        err = np.linalg.norm(pred - B, axis=1)
        inl = err <= tol_frac + EPS
        inl_sum = int(inl.sum())
        # criterio deterministico: più inlier; a parità, mediana errore minore; a parità, indici inlier less-lex
        if inl_sum > best_sum:
            best_sum = inl_sum
            best_inl = inl
            best_med = float(np.median(err[inl])) if inl.any() else None
        elif inl_sum == best_sum and inl_sum > 0:
            med_err = float(np.median(err[inl]))
            if best_med is None or med_err < best_med - EPS:
                best_inl = inl
                best_med = med_err
    return best_inl, best_med

# ======= PIPELINE =======
@torch.inference_mode()
def validate_tiled_pair(slide_path: str, block_path: str) -> Dict[str, Any]:
    set_strict_determinism()
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Preprocess → PIL 512×512 allineate
    slide_pil, _ = prepare_image_for_snn(slide_path, out_size=INPUT_SIZE, return_debug=False)
    block_pil, _ = prepare_image_for_snn(block_path, out_size=INPUT_SIZE, return_debug=False)

    # (facoltativo) checksum per verificare identicità input tra run
    def sha(a: np.ndarray) -> str: return hashlib.sha256(a.tobytes()).hexdigest()
    # print("CHK slide:", sha(np.array(slide_pil)), "CHK block:", sha(np.array(block_pil)))

    # 2) Tiles multiscala (solo tessuto)
    slide_tiles, slide_boxes = extract_tissue_tiles_multiscale(slide_pil, TILE_SCALES, STRIDE_FACTOR, TILE_MIN_TISSUE)
    block_tiles, block_boxes = extract_tissue_tiles_multiscale(block_pil, TILE_SCALES, STRIDE_FACTOR, TILE_MIN_TISSUE)
    Ns, Nb = len(slide_tiles), len(block_tiles)
    if Ns == 0 or Nb == 0:
        return {"same_tissue": False, "reason": "no_tissue_tiles", "Ns": Ns, "Nb": Nb}

    # 3) Orientazioni (rot + flip) e embedding
    slide_views = augment_orientations(slide_tiles)
    block_views = augment_orientations(block_tiles)
    k = len(slide_views[0])

    model = EmbeddingNet(dim_out=256).to(device).eval()

    Es = embed_oriented_tiles(model, slide_views)     # (Ns*k, D)
    Eb = embed_oriented_tiles(model, block_views)     # (Nb*k, D)

    # Self-similarity su UNA vista per tile (prima rotazione), per gap relativo
    Es_one = model.embed_pils([v[0] for v in slide_views])  # (Ns, D)
    Eb_one = model.embed_pils([v[0] for v in block_views])  # (Nb, D)
    selfS = self_cosine_stats(Es_one); selfB = self_cosine_stats(Eb_one)
    self_med_min = min(selfS["med"], selfB["med"])
    self_p25_min = min(selfS["p25"], selfB["p25"])

    # 4) Matrice distanze minime per coppia di tile (tutte le orientazioni)
    D = oriented_min_dist_matrix(Es, Eb, Ns, Nb, k)

    # 5) Matching simmetrico kNN+ratio (deterministico)
    matches_ab = knn_ratio_matches(D, k=KNN_K, ratio_max=RATIO_MAX, abs_min=ABS_COS_THR_MIN)
    matches_ba = knn_ratio_matches(D.T, k=KNN_K, ratio_max=RATIO_MAX, abs_min=ABS_COS_THR_MIN)
    prec_ab, prec_ba, overlap_j = symmetric_consistency(matches_ab, matches_ba)
    matches = matches_ab

    if len(matches) == 0:
        return {"same_tissue": False, "reason": "no_knn_matches", "Ns": Ns, "Nb": Nb}

    # 6) Statistiche distanze
    dists = np.array([m[2] for m in matches], np.float32)
    mn, med = float(dists.min()), float(np.median(dists))

    # 7) Copertura e dispersione
    used_slide = {i for i, _, _ in matches}
    cover_tiles = len(used_slide) / max(Ns, 1)
    disp_quads = quadrant_dispersion(slide_boxes, list(used_slide), size=INPUT_SIZE)

    # 8) Geometria: RANSAC deterministico
    cent_s = [(slide_boxes[i][0] + slide_boxes[i][2]*0.5,
               slide_boxes[i][1] + slide_boxes[i][3]*0.5) for i, _, _ in matches]
    cent_b = [(block_boxes[j][0] + block_boxes[j][2]*0.5,
               block_boxes[j][1] + block_boxes[j][3]*0.5) for _, j, _ in matches]
    inl_mask, med_err_norm = ransac_similarity_norm(cent_s, cent_b,
                                                    iters=RANSAC_ITERS,
                                                    tol_frac=RANSAC_TOL_FRAC,
                                                    seed=RANSAC_SEED)
    n_inl = int(inl_mask.sum())
    inl_ratio = n_inl / max(len(matches), 1)

    # 9) Paletti adattivi
    factor = 0.08
    if med < 0.15 and mn < 0.10:
        factor = 0.07
    min_matches = max(MIN_MATCHES_FLOOR, int(factor * Ns))
    min_inliers = max(MIN_INLIERS_FLOOR, int(0.30 * min_matches))

    # 10) Doppio cancello: assoluto + relativo alla self-similarity
    ok_dists_abs = (mn <= ABS_COS_THR_MIN + EPS) and (med <= ABS_COS_THR_MED + EPS)
    GAP_COS_MIN = 0.25
    GAP_COS_MED = 0.30
    ok_dists_rel = (mn <= self_p25_min + GAP_COS_MIN + EPS) and (med <= self_med_min + GAP_COS_MED + EPS)

    # 11) Vincoli restanti
    ok_cover = (cover_tiles >= MIN_TILE_COVERAGE - EPS)
    ok_count = (len(matches) >= min_matches)
    ok_geom  = (n_inl >= min_inliers) and (inl_ratio >= MIN_INLIER_RATIO - EPS)
    ok_sym = (prec_ab >= 0.30 and prec_ba >= 0.30) or (overlap_j >= 0.10)
    ok_disp  = (disp_quads >= 3)

    same = bool(ok_dists_abs and ok_dists_rel and ok_cover and ok_count and ok_geom and ok_sym and ok_disp)

    return {
        "same_tissue": same,
        "slide": slide_path, "block": block_path,
        "Ns": Ns, "Nb": Nb,
        "matches": len(matches),
        "tile_coverage_slide": round(cover_tiles, 3),
        "dist_min": round(mn, 4), "dist_med": round(med, 4),
        "self_slide_med": round(selfS["med"], 4), "self_block_med": round(selfB["med"], 4),
        "self_slide_p25": round(selfS["p25"], 4), "self_block_p25": round(selfB["p25"], 4),
        "overlap_jaccard": round(overlap_j, 3),
        "quadrant_dispersion": disp_quads,
        "ransac_inliers": n_inl, "inlier_ratio": round(inl_ratio, 3),
        "ransac_med_err_norm": None if med_err_norm is None else round(med_err_norm, 4),
        "sym_precision_ab": round(prec_ab, 3),
        "sym_precision_ba": round(prec_ba, 3),
        "adaptive_requirements": {
            "min_matches": min_matches,
            "min_inliers": min_inliers,
            "min_tile_coverage": MIN_TILE_COVERAGE
        },
        "thresholds": {
            "cos_min": ABS_COS_THR_MIN, "cos_med": ABS_COS_THR_MED,
            "ratio_max": RATIO_MAX, "ransac_tol_frac": RANSAC_TOL_FRAC
        }
    }

# ======= MAIN =======
if __name__ == "__main__":
    set_strict_determinism()
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Deterministic: ON")
    out = validate_tiled_pair(SLIDE_PATH, BLOCK_PATH)
    print("\n=== SNN tiled validation (deterministic) ===")
    print(out)
