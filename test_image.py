# evaluate_davis_segdino.py
# Evaluate SegDINO (frame-by-frame) on DAVIS 2017 val set
# Clean output: one line per sequence + final summary

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms as T

# Suppress OpenCV and libpng warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
cv2.setLogLevel(0)

# -------------------- Metrics --------------------
def compute_mae(pred, gt):
    pred = np.clip(pred, 0.0, 1.0)
    gt_norm = np.clip(gt / 255.0, 0.0, 1.0)
    return float(np.mean(np.abs(pred - gt_norm)))


def compute_fmeasure(pred_list, gt_list, cuda=True):
    """Exact replication of Eval_fmeasure from evaluator_vsod.py"""
    beta2 = 0.3  # Hard-coded exactly as in original code
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # Convert lists to tensors on device
    pred_tensors = [torch.from_numpy(p).to(device) if isinstance(p, np.ndarray) else p.to(device) 
                    for p in pred_list]
    gt_tensors = [torch.from_numpy(g).to(device) if isinstance(g, np.ndarray) else g.to(device) 
                  for g in gt_list]

    # We only need ToTensor for PIL/ndarray → but since we already have tensors, we skip it
    # Instead, we ensure GT is float [0,1] if needed

    avg_f = torch.zeros(255, device=device)
    img_num = 0.0

    with torch.no_grad():
        for pred, gt in zip(pred_tensors, gt_tensors):
            # Normalize prediction to [0,1]
            pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)

            # Ensure GT is float tensor in [0,1]
            if gt.max() > 1.0:
                gt = gt.float() / 255.0   # if it was uint8 0-255
            else:
                gt = gt.float()

            # Now both are tensors on device → compute PR
            prec, recall = _eval_pr(pred, gt, 255)

            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for NaN

            avg_f += f_score
            img_num += 1.0

    if img_num > 0:
        Fm = avg_f / img_num
        max_f = Fm.max().item()
    else:
        max_f = 0.0

    return max_f


def _eval_pr(y_pred, y, num):
    prec = torch.zeros(num, device=y_pred.device)
    recall = torch.zeros(num, device=y_pred.device)
    thlist = torch.linspace(0, 1 - 1e-10, num, device=y_pred.device)

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i] = tp / (y_temp.sum() + 1e-20)
        recall[i] = tp / (y.sum() + 1e-20)

    return prec, recall


def compute_smeasure(pred, gt):
    alpha = 0.5
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt_norm = np.clip(gt.astype(np.float32) / 255.0, 0.0, 1.0)
    y = gt_norm.mean()
    if y < 1e-4:
        return float(1.0 - pred.mean())
    elif y > 1 - 1e-4:
        return float(pred.mean())
    else:
        gt_bin = (gt_norm >= 0.5).astype(np.float32)
        Q_obj = _object(pred, gt_bin)
        Q_reg = _region(pred, gt_bin)
        S = alpha * Q_obj + (1 - alpha) * Q_reg
        return float(np.clip(S, 0.0, 1.0))


def _object(pred, gt):
    fg = pred[gt == 1]
    bg = 1 - pred[gt == 0]

    def score(x):
        if x.size == 0:
            return 0.0
        mu = x.mean()
        sigma = x.std()
        return 2 * mu / (mu**2 + 1 + sigma + 1e-8)

    w_fg = gt.mean()
    w_bg = 1 - w_fg
    return w_fg * score(fg) + w_bg * score(bg)


def _region(pred, gt):
    h, w = gt.shape
    X, Y = _centroid(gt)

    p1, g1 = pred[:X, :Y], gt[:X, :Y]
    p2, g2 = pred[:X, Y:], gt[:X, Y:]
    p3, g3 = pred[X:, :Y], gt[X:, :Y]
    p4, g4 = pred[X:, Y:], gt[X:, Y:]

    area1 = X * Y
    area2 = X * (w - Y)
    area3 = (h - X) * Y
    area4 = (h - X) * (w - Y)

    Sr = (
        area1 * _ssim(p1, g1) +
        area2 * _ssim(p2, g2) +
        area3 * _ssim(p3, g3) +
        area4 * _ssim(p4, g4)
    ) / (h * w + 1e-8)

    return Sr


def _centroid(gt):
    h, w = gt.shape
    if gt.sum() == 0:
        return h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    total = gt.sum()
    X = int(np.round((yy * gt).sum() / total))
    Y = int(np.round((xx * gt).sum() / total))
    return X, Y


def _ssim(pred, gt):
    h, w = pred.shape
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = np.sum((pred - x) ** 2) / (N - 1 + 1e-20)
    sigma_y2 = np.sum((gt - y) ** 2) / (N - 1 + 1e-20)
    sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1 + 1e-20)
    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        Q = alpha / (beta + 1e-20)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0.0
    return Q


def compute_emax(pred: np.ndarray, gt: np.ndarray, num_thresholds: int = 255) -> float:
    """
    Compute maximum Enhanced-alignment measure (E_m) over thresholds in [0,1].

    Parameters
    ----------
    pred : np.ndarray
        HxW probability map (float). Values are clipped to [0, 1].
    gt : np.ndarray
        HxW ground-truth mask (uint8 or float). If max(gt) > 1, it is assumed to be 0..255
        and will be scaled to [0,1]. Then binarized at 0.5.
    num_thresholds : int
        Number of thresholds in [0,1] to evaluate (default 255).

    Returns
    -------
    float
        E_max (maximum E-measure across thresholds).
    """
    # --- sanitize inputs ---
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.clip(pred, 0.0, 1.0)

    gt = np.asarray(gt)
    if not np.issubdtype(gt.dtype, np.floating):
        gt = gt.astype(np.float32)
    if gt.max() > 1.0:  # e.g., uint8 mask in {0,255}
        gt = gt / 255.0
    gt_bin = (gt >= 0.5).astype(np.float32)

    eps = 1e-8
    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    def e_measure_binary(pb: np.ndarray, gb: np.ndarray) -> float:
        """
        Canonical E-measure for two *binary* maps in {0,1}.
        """
        p_mean = pb.mean(dtype=np.float64)
        g_mean = gb.mean(dtype=np.float64)

        # Degenerate GT handling (common convention)
        if g_mean == 0.0:
            return float(1.0 - p_mean)  # best is predict all background
        if g_mean == 1.0:
            return float(p_mean)        # best is predict all foreground

        p_c = pb - p_mean
        g_c = gb - g_mean
        denom = (p_c * p_c) + (g_c * g_c) + eps
        align = (2.0 * p_c * g_c) / denom
        phi = ((1.0 + align) ** 2) * 0.25
        return float(phi.mean())

    # Sweep thresholds, compute E for each, take max
    e_max = 0.0
    for t in thresholds:
        pred_bin = (pred >= t).astype(np.float32)
        e_val = e_measure_binary(pred_bin, gt_bin)
        if e_val > e_max:
            e_max = e_val

    return float(e_max)



# -------------------- Model Loading --------------------
def load_model(ckpt_path, dino_ckpt_path, repo_dir, dino_size, device):
    if dino_size == "b":
        backbone = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=dino_ckpt_path)
    else:
        backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt_path)

    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone)
    model = model.to(device)

    print(f"[Loading segmentation checkpoint] {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    state = obj['state_dict'] if 'state_dict' in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

    return model


# -------------------- Main Evaluation --------------------
# -------------------- Main Evaluation --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegDINO on DAVIS 2017 val set (clean output)")
    parser.add_argument("--davis_root", type=str, default="/home/raed/Documents/datasets_local/DUTS-TE",
                        help="Root path to DAVIS dataset")
    parser.add_argument("--ckpt", type=str, default="./ckpt_path",
                        help="Path to trained SegDINO checkpoint (.pth)")
    parser.add_argument("--dino_ckpt", type=str, default="./dinov3_vits16_pretrain.pth",
                        help="Path to pretrained DINOv3 weights (.pth)")
    parser.add_argument("--repo_dir", type=str, default="./dinov3",
                        help="Local path to DINOv3 torch.hub repo")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size")
    parser.add_argument("--input_h", type=int, default=448)
    parser.add_argument("--input_w", type=int, default=448)
    parser.add_argument("--output_dir", type=str, default="./eval_davis_segdino")
    parser.add_argument("--dice_thr", type=float, default=0.5)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        ckpt_path=args.ckpt,
        dino_ckpt_path=args.dino_ckpt,
        repo_dir=args.repo_dir,
        dino_size=args.dino_size,
        device=device
    )
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    # ────────────────────────────────────────────────────────────────
    # IMAGE DATASET LOADING (MODIFIED PART)
    # ────────────────────────────────────────────────────────────────
    img_root = os.path.join(args.davis_root, "images")
    gt_root  = os.path.join(args.davis_root, "masks")

    img_files = sorted([f for f in os.listdir(img_root)
                        if f.lower().endswith(('.jpg', '.png'))])

    print(f"Found {len(img_files)} images\n")

    pred_list, gt_list = [], []

    for name in tqdm(img_files, desc="Evaluating images"):
        img_path = os.path.join(img_root, name)
        gt_path  = os.path.join(gt_root, os.path.splitext(name)[0] + ".png")

        if not os.path.exists(gt_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img_resized = cv2.resize(img, (args.input_w, args.input_h))
        img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1)
        img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor))

        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
        pred = prob.squeeze().cpu().numpy()

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        pred_list.append(pred)
        gt_list.append(gt)

    # ────────────────────────────────────────────────────────────────
    # METRICS (UNCHANGED)
    # ────────────────────────────────────────────────────────────────
    if not pred_list:
        print("No valid image-mask pairs found.")
        exit(0)

    fbeta = compute_fmeasure(pred_list, gt_list)
    smeasure = np.mean([compute_smeasure(p, g) for p, g in zip(pred_list, gt_list)])
    mae = np.mean([compute_mae(p, g) for p, g in zip(pred_list, gt_list)])
    emax = np.mean([compute_emax(p, g) for p, g in zip(pred_list, gt_list)])

    print("\n" + "=" * 60)
    print("FINAL RESULTS — IMAGE DATASET")
    print("=" * 60)
    print(f"S-measure     : {smeasure:.4f}")
    print(f"Max F-β       : {fbeta:.4f}")
    print(f"MAE           : {mae:.4f}")
    print(f"Max E-measure : {emax:.4f}")
    print("=" * 60)

    np.save(os.path.join(args.output_dir, "image_dataset_metrics.npy"), {
        "smeasure": smeasure,
        "fbeta_max": fbeta,
        "mae": mae,
        "emax": emax
    })

