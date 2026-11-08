import os
import sys
import time
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score

try:
    from models import (
        LateFusionModel, GatedConcatFusionModel, GatedSumFusionModel,
    )
except ImportError:
    from src.models import (
        LateFusionModel, GatedConcatFusionModel, GatedSumFusionModel,
    )


# =============================================================================
# Config
# =============================================================================
# Central defaults for data, training, augmentation, splitting, threshold
# tuning, and latency measurement. The values for p_mask, p_corrupt, and
# sigma_max control training-time robustness augmentation; test-time conditions
# are specified separately below.
DEFAULT_CONFIG = {
    "feature_path": "data/processed/extracted_features.pt",
    "output_dir": "experiments/results",
    "ckpt_dir": "experiments/checkpoints",
    # Number of optimizer passes over the training split, minibatch size, and
    # Adam learning rate, respectively.
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    # Training augmentation rates: p_mask allocates a sample to image- or
    # text-missing cases; p_corrupt controls image corruption probability
    # (text uses half that probability); sigma_max bounds sampled noise scale.
    "p_mask": 0.10,          # was 0.15
    "p_corrupt": 0.15,       # was 0.50  ← the critical change
    "sigma_max": 0.15,       # was 0.40  ← the critical change
    # Fractions of the shuffled feature rows assigned to train and validation;
    # test receives the remainder.
    "train_frac": 0.70,
    "val_frac": 0.15,
    # Multiplier on each auxiliary-head BCE loss for learned gated models.
    "aux_weight": 0.3,
    # Retained in config for reference; model construction in this file does
    # not pass it, so classifier layers use their class-defined default.
    "hidden_dim": 256,
    # Candidate global decision thresholds; the best macro-F1 on clean
    # validation predictions is selected and reused on all test conditions.
    "thresholds": tuple(np.round(np.arange(0.10, 0.91, 0.05), 2)),
    # Number of initial test batches excluded from forward-latency timing.
    "latency_warmup_batches": 5,
}

# Test-time conditions passed to inference. A mask marks a modality unavailable
# and is handled by the model; a positive sigma adds Gaussian noise to that
# feature vector before the forward pass. The listed corruption sweep changes
# only image features; text sigma remains zero in all these named conditions.
# The oracle model uses these mask/noise settings to calculate its fixed weights.
CONDITIONS = {
    "clean":               {"mask_image": False, "mask_text": False, "sigma_img": 0.00, "sigma_txt": 0.0},
    "image_missing":       {"mask_image": True,  "mask_text": False, "sigma_img": 0.00, "sigma_txt": 0.0},
    "text_missing":        {"mask_image": False, "mask_text": True,  "sigma_img": 0.00, "sigma_txt": 0.0},
    "image_corrupt_s0.05": {"mask_image": False, "mask_text": False, "sigma_img": 0.05, "sigma_txt": 0.0},
    "image_corrupt_s0.10": {"mask_image": False, "mask_text": False, "sigma_img": 0.10, "sigma_txt": 0.0},
    "image_corrupt_s0.20": {"mask_image": False, "mask_text": False, "sigma_img": 0.20, "sigma_txt": 0.0},
    "image_corrupt_s0.30": {"mask_image": False, "mask_text": False, "sigma_img": 0.30, "sigma_txt": 0.0},
    "image_corrupt_s0.50": {"mask_image": False, "mask_text": False, "sigma_img": 0.50, "sigma_txt": 0.0},
}

MODEL_SPECS = {
    # Each entry selects a fusion architecture and whether training-time
    # augmentation or condition-derived oracle weights are used. The
    # "moddrop" name refers to augmentation in train_one_model, not a separate
    # layer inside LateFusionModel.
    "A_latefusion_noaug":     {"cls": LateFusionModel,         "aug": False, "oracle": False},
    "B_latefusion_moddrop":   {"cls": LateFusionModel,         "aug": True,  "oracle": False},
    "C_gated_concat":         {"cls": GatedConcatFusionModel,  "aug": True,  "oracle": False},
    "D_oracle_gated_concat":  {"cls": GatedConcatFusionModel,  "aug": True,  "oracle": True},
    # Optional: uncomment to also test the weighted-sum variant
    # "E_gated_sum":          {"cls": GatedSumFusionModel,     "aug": True,  "oracle": False},
}


# =============================================================================
# Metrics
# =============================================================================
def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    # Convert independent class probabilities to binary decisions with one
    # shared threshold. F1 is reported both macro-averaged (equal weight per
    # class) and micro-averaged (aggregate decisions across all classes).
    y_pred = (y_pred_probs >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Average precision uses the full probability ranking, not thresholded
    # predictions. Skip classes with no positive examples because AP is not
    # informative for them; the returned mAP is the mean over remaining
    # classes. Precision and recall are not returned as separate metrics here.
    ap_scores, n_skipped = [], 0
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            ap_scores.append(float(average_precision_score(y_true[:, c], y_pred_probs[:, c])))
        else:
            n_skipped += 1

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "map": float(np.mean(ap_scores)) if ap_scores else 0.0,
        "map_n_classes_used": len(ap_scores),
        "map_n_classes_skipped": n_skipped,
        "threshold": float(threshold),
    }


# =============================================================================
# Oracle reliability weights
# =============================================================================
def _reliability(sigma):
    # Map the noise standard deviation to a reliability score in (0, 1]:
    # clean input (sigma=0) gets 1, and larger noise lowers the score
    # exponentially. This heuristic supplies oracle weights; it is not learned.
    return float(np.exp(-3.0 * sigma))


def oracle_weights_from_condition(mask_image, mask_text, sigma_img, sigma_txt, B, device):
    """Convert one test condition into repeated, normalized (B, 2) gate weights."""
    # A missing modality has zero reliability; otherwise reliability depends
    # on its configured noise level. Normalize the two scores to sum to one.
    r_img = (0.0 if mask_image else _reliability(sigma_img))
    r_txt = (0.0 if mask_text else _reliability(sigma_txt))
    if r_img + r_txt < 1e-9:
        # If both inputs are unavailable, use equal weights as a safe fallback.
        return torch.full((B, 2), 0.5, device=device)
    tot = r_img + r_txt
    return torch.tensor([[r_img / tot, r_txt / tot]] * B, device=device, dtype=torch.float32)


# =============================================================================
# Inference (returns probabilities; also collects gate weights if available)
# =============================================================================
def run_inference(model, dataloader, device, condition, use_oracle,
                  renormalize_after_corruption=False):
    # Run a trained fusion model on the supplied data split under one condition
    # (clean validation data is also passed here for threshold tuning).
    # Returns per-class probabilities and targets, plus summary gate weights
    # when the architecture exposes them. No gradients are computed.
    model.eval()
    mask_image = condition["mask_image"]
    mask_text = condition["mask_text"]
    sigma_img = condition["sigma_img"]
    sigma_txt = condition["sigma_txt"]

    all_probs, all_targets = [], []
    all_w_img, all_w_txt = [], []

    with torch.no_grad():
        for img_f, txt_f, lbls in dataloader:
            # Each dataloader row is a precomputed image embedding, text
            # embedding, and multi-hot target vector, kept aligned by dataset
            # index. Move model inputs to the device; these CPU-loaded targets
            # are left on CPU for NumPy-based metric calculation.
            img_f = img_f.to(device)
            txt_f = txt_f.to(device)
            B = img_f.size(0)

            if sigma_img > 0.0:
                # Corruption is additive, zero-mean Gaussian noise applied
                # independently to every feature component with the requested
                # standard deviation. By default vectors are not renormalized
                # afterward, so noise can also change their L2 magnitude.
                img_f = img_f + torch.randn_like(img_f) * sigma_img
                if renormalize_after_corruption:
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if sigma_txt > 0.0:
                txt_f = txt_f + torch.randn_like(txt_f) * sigma_txt
                if renormalize_after_corruption:
                    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            oracle_w = None
            if use_oracle:
                # For the oracle variant, condition-level reliability weights
                # are passed to the model instead of relying on its learned
                # gate. The same condition applies to all samples in this run.
                oracle_w = oracle_weights_from_condition(
                    mask_image, mask_text, sigma_img, sigma_txt, B, device
                )

            logits, aux = model(
                img_f, txt_f,
                mask_image=mask_image, mask_text=mask_text,
                oracle_weights=oracle_w,
            )
            # Sigmoid turns each class logit into an independent probability;
            # multi-label outputs are not a mutually exclusive softmax.
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(lbls.numpy())
            if "w_image" in aux and aux["w_image"] is not None:
                all_w_img.append(aux["w_image"].cpu().numpy())
                all_w_txt.append(aux["w_text"].cpu().numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)
    gate = {}
    if all_w_img:
        # Summarize the per-sample gate weights over the full inference split.
        w_i = np.concatenate(all_w_img)
        w_t = np.concatenate(all_w_txt)
        gate = {
            "w_image_mean": float(w_i.mean()),
            "w_image_std": float(w_i.std()),
            "w_text_mean": float(w_t.mean()),
            "w_text_std": float(w_t.std()),
        }
    return probs, targets, gate


def measure_latency(model, dataloader, device, warmup_batches=5, use_oracle=False):
    """Measure plain forward-pass latency after warmup, in milliseconds/sample."""
    model.eval()

    # Warmup reduces the effect of one-time setup costs. This benchmark calls
    # the model with only feature inputs: it does not apply masks, corruption,
    # oracle weights, or the use_oracle argument.
    with torch.no_grad():
        for i, (img_f, txt_f, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            img_f, txt_f = img_f.to(device), txt_f.to(device)
            _ = model(img_f, txt_f)

    if device == "cuda":
        torch.cuda.synchronize()

    n, start = 0, time.perf_counter()
    with torch.no_grad():
        for img_f, txt_f, _ in dataloader:
            img_f, txt_f = img_f.to(device), txt_f.to(device)
            _ = model(img_f, txt_f)
            n += img_f.size(0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(elapsed / max(n, 1) * 1000.0)


def tune_threshold(y_true, y_pred_probs, thresholds):
    # Select one global threshold from the configured candidate values by
    # maximizing macro-F1 on validation data. Strict '>' keeps the first
    # threshold in the configured ordering in the event of a tie.
    best = (0.5, -1.0)
    for t in thresholds:
        f1 = f1_score(y_true, (y_pred_probs >= t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best[0], best[1]


# =============================================================================
# Augmentation & training
# =============================================================================
def augment_batch(img_f, txt_f, device, p_mask=0.15, p_corrupt=0.5, sigma_max=0.4):
    """
    Returns: img_aug, txt_aug, mask_img (B,), mask_txt (B,),
             sigma_img (B,), sigma_txt (B,)
    sigma_img/txt are the per-sample noise std actually applied (0 if none).
    """
    # Training-time robustness augmentation for a batch of cached features.
    # The returned masks and per-sample sigmas describe what was sampled and
    # are used to inform the model (and optionally the oracle gate).
    B = img_f.size(0)
    r = torch.rand(B, device=device)
    # A single random draw assigns at most one modality mask per sample:
    # image for r < p_mask, text for p_mask <= r < 2*p_mask, neither otherwise.
    mask_img = r < p_mask
    mask_txt = (r >= p_mask) & (r < 2 * p_mask)

    # Noise corruption is sampled independently of those masks. Therefore a
    # feature can receive noise and also be marked unavailable; the model's
    # mask then zeros it before fusion, so that corrupted value is not used.
    # Independently decide which samples receive image noise, then draw each
    # selected sample's sigma uniformly from [0, sigma_max). Multiplying by
    # the flag makes sigma exactly zero for uncorrupted samples.
    corrupt_img_flag = torch.rand(B, device=device) < p_corrupt
    sigma_img = torch.rand(B, device=device) * sigma_max * corrupt_img_flag.float()
    img_aug = img_f + torch.randn_like(img_f) * sigma_img.unsqueeze(-1)

    # Text corruption is sampled less often (half p_corrupt) and has half the
    # maximum noise scale. Noise is added to features, not raw text/tokens.
    corrupt_txt_flag = torch.rand(B, device=device) < (p_corrupt * 0.5)
    sigma_txt = torch.rand(B, device=device) * (sigma_max * 0.5) * corrupt_txt_flag.float()
    txt_aug = txt_f + torch.randn_like(txt_f) * sigma_txt.unsqueeze(-1)

    return img_aug, txt_aug, mask_img, mask_txt, sigma_img, sigma_txt


def oracle_weights_from_batch(mask_img, mask_txt, sigma_img, sigma_txt, device):
    """Build per-sample oracle weights from the augmentation actually sampled."""
    # The reliability decreases with the applied per-sample noise, and a
    # masked modality gets zero reliability. Normalize the pair to produce
    # weights compatible with the model's two modality gates.
    r_img = (~mask_img).float() * torch.exp(-3.0 * sigma_img)
    r_txt = (~mask_txt).float() * torch.exp(-3.0 * sigma_txt)
    tot = (r_img + r_txt).clamp_min(1e-8)
    fallback = (r_img + r_txt) < 1e-9
    w_i = torch.where(fallback, torch.full_like(r_img, 0.5), r_img / tot)
    w_t = torch.where(fallback, torch.full_like(r_txt, 0.5), r_txt / tot)
    return torch.stack([w_i, w_t], dim=-1)


def train_one_model(model, train_loader, device, cfg, use_aug, use_oracle):
    # Fit the fusion/classification layers on training features. The CLIP
    # encoders are not involved here; input tensors are already extracted.
    model.train()
    # Adam updates this model's parameters. BCEWithLogitsLoss treats each class
    # as an independent binary target and combines sigmoid with binary cross
    # entropy in a numerically stable loss.
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.BCEWithLogitsLoss()
    loss_history = []

    # Auxiliary per-modality losses are enabled only for the learned gated
    # architecture. For the oracle variant, fixed weights do not create a
    # gradient path through the gate, so these auxiliary heads are not used.
    has_aux = hasattr(model, "aux_img") and not use_oracle
    aux_weight = cfg.get("aux_weight", 0.3)

    for epoch in range(cfg["epochs"]):
        epoch_loss, n_batches = 0.0, 0
        for img_f, txt_f, lbls in train_loader:
            img_f, txt_f, lbls = img_f.to(device), txt_f.to(device), lbls.to(device)
            B = img_f.size(0)
            oracle_w = None
            mask_img = torch.zeros(B, dtype=torch.bool, device=device)
            mask_txt = torch.zeros(B, dtype=torch.bool, device=device)

            if use_aug:
                # The "moddrop" training variant uses sampled availability
                # masks and additive feature noise. Masks are passed to the
                # model, which zeros unavailable modalities; augmentation
                # itself does not delete dataset rows.
                img_f, txt_f, mask_img, mask_txt, s_img, s_txt = augment_batch(
                    img_f, txt_f, device,
                    p_mask=cfg["p_mask"],
                    p_corrupt=cfg["p_corrupt"],
                    sigma_max=cfg["sigma_max"],
                )
                if use_oracle:
                    # The oracle receives weights derived from this batch's
                    # actual masks/noise rather than a learned gate output.
                    oracle_w = oracle_weights_from_batch(
                        mask_img, mask_txt, s_img, s_txt, device
                    )

            optimizer.zero_grad()
            if has_aux:
                # Train both the fused classifier and the two modality heads.
                # The auxiliary losses encourage each gated modality to retain
                # predictive signal; aux_weight controls their contribution.
                logits, aux = model(
                    img_f, txt_f,
                    mask_image=mask_img, mask_text=mask_txt,
                    oracle_weights=oracle_w, return_aux=True,
                )
                loss = criterion(logits, lbls)
                loss = loss + aux_weight * criterion(aux["img_logits"], lbls)
                loss = loss + aux_weight * criterion(aux["txt_logits"], lbls)
            else:
                logits, _ = model(
                    img_f, txt_f,
                    mask_image=mask_img, mask_text=mask_txt,
                    oracle_weights=oracle_w,
                )
                loss = criterion(logits, lbls)

            # Backpropagate the selected objective and update the model weights.
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)
        print(f"    epoch {epoch + 1:02d}/{cfg['epochs']} | loss={avg:.4f}")

    return loss_history


# =============================================================================
# Data loading & splitting
# =============================================================================
def load_and_split(cfg, seed, device):
    # Read the artifact produced by features.py. Its image/text embedding rows
    # and labels share a common sample order; apply one permutation to all
    # three tensors so their pairings remain intact.
    data = torch.load(cfg["feature_path"], weights_only=False)
    img_emb = data["image_embeddings"]
    txt_emb = data["text_embeddings"]
    labels = data["labels"]
    n = img_emb.size(0)
    print(f"Loaded {n} samples, {labels.size(1)} classes from {cfg['feature_path']}")

    # A local seeded generator makes the randomized, non-stratified split
    # repeatable without depending on unrelated global random draws.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    img_emb, txt_emb, labels = img_emb[perm], txt_emb[perm], labels[perm]

    # Use configured fractions for train and validation; the remaining rows
    # form the test split. TensorDataset yields aligned (image, text, label)
    # tuples for the training and inference DataLoaders.
    n_train = int(n * cfg["train_frac"])
    n_val = int(n * cfg["val_frac"])
    tr = TensorDataset(img_emb[:n_train], txt_emb[:n_train], labels[:n_train])
    va = TensorDataset(img_emb[n_train:n_train + n_val],
                       txt_emb[n_train:n_train + n_val],
                       labels[n_train:n_train + n_val])
    te = TensorDataset(img_emb[n_train + n_val:], txt_emb[n_train + n_val:], labels[n_train + n_val:])
    print(f"Split: train={len(tr)} | val={len(va)} | test={len(te)}")
    return tr, va, te


# =============================================================================
# Per-seed pipeline
# =============================================================================
def run_seed(cfg, seed, device, quick=False):
    # Execute one full replicate: reset random sources, split the feature set,
    # train every configured model, select thresholds, and evaluate conditions.
    print(f"\n{'=' * 70}\nSeed {seed}\n{'=' * 70}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if quick:
        # Avoid mutating the caller's shared config when reducing a smoke run
        # to one epoch.
        cfg = copy.deepcopy(cfg)
        cfg["epochs"] = 1

    # Shuffle training rows each epoch; keep validation/test iteration order
    # deterministic. The same batch structure serves all model variants.
    tr, va, te = load_and_split(cfg, seed, device)
    bs = cfg["batch_size"]
    train_loader = DataLoader(tr, batch_size=bs, shuffle=True)
    val_loader = DataLoader(va, batch_size=bs, shuffle=False)
    test_loader = DataLoader(te, batch_size=bs, shuffle=False)

    num_classes = tr.tensors[2].size(1)
    seed_results = {}
    seed_losses = {}

    for model_name, spec in MODEL_SPECS.items():
        print(f"\n  >>> {model_name}")
        # Instantiate and train a fresh model for each architecture/variant;
        # no trained checkpoint is loaded by this pipeline.
        model = spec["cls"](num_classes=num_classes).to(device)
        losses = train_one_model(
            model, train_loader, device, cfg,
            use_aug=spec["aug"], use_oracle=spec["oracle"],
        )
        seed_losses[model_name] = losses

        # ---- Tune threshold on val (clean) ----
        # Select the operating threshold on clean validation probabilities,
        # then reuse it for every test condition for a fair comparison.
        torch.manual_seed(seed * 1000 + 0)  # deterministic noise draws
        val_probs, val_true, _ = run_inference(
            model, val_loader, device, CONDITIONS["clean"], use_oracle=spec["oracle"],
        )
        best_t, best_f1 = tune_threshold(val_true, val_probs, cfg["thresholds"])
        print(f"    best val threshold={best_t:.2f} | val macro-F1={best_f1:.4f}")

        # ---- Save checkpoint ----
        # Store this seed/model's trained parameters for later reuse.
        os.makedirs(cfg["ckpt_dir"], exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(cfg["ckpt_dir"], f"seed{seed}_{model_name}.pt"),
        )

        # ---- Evaluate on test under every condition ----
        # Reset the random stream by condition index so corresponding
        # conditions use reproducible noise draws across model variants.
        seed_results[model_name] = {}
        for cond_idx, (cond_name, cond) in enumerate(CONDITIONS.items()):
            # Same seed for all models -> identical noise tensors per condition
            torch.manual_seed(seed * 1000 + 100 + cond_idx)
            test_probs, test_true, gate = run_inference(
                model, test_loader, device, cond, use_oracle=spec["oracle"],
            )
            metrics = calculate_metrics(test_true, test_probs, threshold=best_t)
            if gate:
                metrics["gate"] = gate
            seed_results[model_name][cond_name] = metrics

        # ---- Latency (clean condition) ----
        # Record forward-only speed and parameter count alongside accuracy
        # metrics; these are model-level values rather than condition scores.
        torch.manual_seed(seed)
        seed_results[model_name]["latency_ms_per_sample"] = measure_latency(
            model, test_loader, device,
            warmup_batches=cfg["latency_warmup_batches"],
            use_oracle=spec["oracle"],
        )
        seed_results[model_name]["num_parameters"] = sum(
            p.numel() for p in model.parameters()
        )

    return seed_results, seed_losses


# =============================================================================
# Aggregation & summary
# =============================================================================
def aggregate_results(all_results):
    """Compute cross-seed mean/std for classification metrics per condition."""
    # Aggregate only macro-F1, micro-F1, and mAP across seeds. Latency is
    # averaged separately; parameter count is constant for a given model.
    agg = {}
    for model_name in MODEL_SPECS:
        agg[model_name] = {}
        for cond_name in CONDITIONS:
            keys = ["macro_f1", "micro_f1", "map"]
            vals = {k: [] for k in keys}
            for seed in all_results:
                r = all_results[seed][model_name][cond_name]
                for k in keys:
                    vals[k].append(r[k])
            agg[model_name][cond_name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in vals.items()
            }
        # Latency & params (constant across conditions, averaged across seeds)
        lat = [all_results[s][model_name]["latency_ms_per_sample"] for s in all_results]
        par = [all_results[s][model_name]["num_parameters"] for s in all_results]
        agg[model_name]["_latency_ms_mean"] = float(np.mean(lat))
        agg[model_name]["_num_parameters"] = int(par[0])
    return agg


def print_summary(agg):
    # Present model complexity/speed once per model and classification metrics
    # as mean ± standard deviation for each named test condition.
    print(f"\n{'=' * 100}\nSUMMARY (mean ± std across seeds)\n{'=' * 100}")
    for model_name in MODEL_SPECS:
        print(f"\n{model_name}  "
              f"(params={agg[model_name]['_num_parameters']:,} | "
              f"{agg[model_name]['_latency_ms_mean']:.4f} ms/sample)")
        header = f"  {'condition':<22} {'macro-F1':<18} {'micro-F1':<18} {'mAP':<18}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cond in CONDITIONS:
            r = agg[model_name][cond]
            print(
                f"  {cond:<22} "
                f"{r['macro_f1']['mean']:.4f}±{r['macro_f1']['std']:.4f}   "
                f"{r['micro_f1']['mean']:.4f}±{r['micro_f1']['std']:.4f}   "
                f"{r['map']['mean']:.4f}±{r['map']['std']:.4f}"
            )


# =============================================================================
# Main
# =============================================================================
def main():
    # CLI options override the defaults most likely to vary between runs:
    # random seeds, training duration, batch size, input feature file, and
    # output location. Other experimental settings remain in DEFAULT_CONFIG.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44",
                        help="comma-separated list of seeds")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--features", default=DEFAULT_CONFIG["feature_path"])
    parser.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: 1 epoch, save to results_quick/")
    args = parser.parse_args()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["feature_path"] = args.features
    # Keep smoke-run artifacts separate to avoid overwriting full results.
    cfg["output_dir"] = args.output_dir + ("_quick" if args.quick else "")
    cfg["ckpt_dir"] = cfg["ckpt_dir"] + ("_quick" if args.quick else "")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    # Training/inference model execution uses CUDA when available and CPU otherwise.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | seeds: {seeds}")

    if not os.path.exists(cfg["feature_path"]):
        # Feature extraction is a separate pipeline stage; fail with a direct
        # instruction rather than attempting to train from absent inputs.
        sys.exit(f"Missing features at {cfg['feature_path']}. Run features.py first.")

    # Run an independent seeded split/training replicate for each requested
    # seed. Retain all per-seed metrics and training-loss histories for output.
    all_results, all_losses = {}, {}
    for seed in seeds:
        all_results[seed], all_losses[seed] = run_seed(cfg, seed, device, quick=args.quick)

    # Save detailed replicate results and loss histories, then the across-seed
    # summary. JSON keys for seeds are strings for serialization.
    with open(os.path.join(cfg["output_dir"], "per_seed_results.json"), "w") as f:
        json.dump({str(s): all_results[s] for s in seeds}, f, indent=2)

    with open(os.path.join(cfg["output_dir"], "training_losses.json"), "w") as f:
        json.dump({str(s): all_losses[s] for s in seeds}, f, indent=2)

    agg = aggregate_results(all_results)
    with open(os.path.join(cfg["output_dir"], "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # Record the effective configuration, conditions, model names, seeds, and
    # selected device beside results so the experiment can be interpreted and
    # reproduced later.
    with open(os.path.join(cfg["output_dir"], "config.json"), "w") as f:
        json.dump(
            {**cfg, "seeds": seeds, "device": device,
             "conditions": CONDITIONS, "models": list(MODEL_SPECS.keys())},
            f, indent=2, default=str,
        )

    print_summary(agg)
    print(f"\nAll artifacts saved to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()