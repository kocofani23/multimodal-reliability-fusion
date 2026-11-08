import os
import sys
import time
import datetime
import argparse
from typing import Optional
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

try:
    from data_loader import COCOMultiModalDataset
except ImportError:
    from src.data_loader import COCOMultiModalDataset


def _as_feature_tensor(output, projection=None, expected_dim=512, tag="feature"):
    """
    Convert a CLIP image/text encoder result into its projected feature tensor.

    Transformers versions may return either the projected tensor directly or
    an output object. In the latter case the object can expose the projected
    image_embeds/text_embeds, or an unprojected pooler_output. For the standard
    openai/clip-vit-base-patch32 model both projected representations have 512
    components. Its vision pooled hidden state is 768-dimensional before the
    visual projection; the text encoder's pre-projection width is 512.

    Handle both output forms, applying the corresponding CLIP projection only
    when the extracted tensor is not already the expected size. Raise an
    explicit error if no tensor is available or the result still has the
    wrong dimensionality.
    """
    if isinstance(output, torch.Tensor):
        feat = output
    else:
        feat = None
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            if hasattr(output, attr):
                candidate = getattr(output, attr)
                if isinstance(candidate, torch.Tensor):
                    feat = candidate
                    break
        if feat is None:
            raise TypeError(
                f"[{tag}] Cannot extract tensor from output of type "
                f"{type(output).__name__}. Attributes: "
                f"{[a for a in dir(output) if not a.startswith('_')]}"
            )

    if feat.dim() == 1:
        feat = feat.unsqueeze(0)

    if feat.shape[-1] != expected_dim and projection is not None:
        feat = projection(feat)

    if feat.shape[-1] != expected_dim:
        raise ValueError(
            f"[{tag}] Expected {expected_dim}-d features, got {feat.shape[-1]}. "
            f"Projection was {'applied' if projection is not None else 'None'}."
        )
    return feat


def _l2norm(x, eps=1e-8):
    # Normalize each final-axis feature vector to unit Euclidean length. The
    # clamp prevents division by zero for a zero vector; this changes scale,
    # not the direction encoded by a nonzero feature.
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


@torch.inference_mode()
def extract_and_save_features(
    image_dir="data/raw/val2017",
    caption_json="data/raw/annotations/captions_val2017.json",
    instance_json="data/raw/annotations/instances_val2017.json",
    output_dir="data/processed",
    model_name="openai/clip-vit-base-patch32",
    batch_size: int = 64,
    seed: int = 42,
    limit: Optional[int] = None,
):
    # Inference mode avoids building autograd graphs or gradients during
    # extraction. Together with the absence of an optimizer/backward pass, the
    # pretrained CLIP weights are used as fixed encoders, not fine-tuned here.
    torch.manual_seed(seed)
    # Create the destination directory if it is not already present.
    os.makedirs(output_dir, exist_ok=True)

    # Prefer CUDA when available; otherwise perform the same extraction on CPU.
    # Both the model and each batch of processor outputs are moved to this
    # device before encoding.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"transformers={transformers.__version__} | torch={torch.__version__}")
    if device.type == "cpu":
        print("[warn] Running on CPU. Extraction for 5k images typically takes "
              "~10-20 minutes with batch_size=64.")

    # Load the pretrained CLIP weights and their matching processor from the
    # same model identifier. CLIP has separate vision and text encoders; the
    # processor provides the model-specific image preprocessing and tokenizer.
    model: CLIPModel = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    # Evaluation mode disables training-specific layer behavior (for example,
    # dropout); it complements inference_mode but does not itself freeze weights.
    model.eval()

    # The dataset connects image files, all captions for each COCO image ID,
    # and its multi-label target. A limit, when supplied, truncates the sorted
    # valid-ID list before batching; it is intended for smaller extraction runs.
    dataset = COCOMultiModalDataset(image_dir, caption_json, instance_json)
    if limit is not None:
        dataset.valid_image_ids = dataset.valid_image_ids[:limit]
        print(f"[info] Limited to {len(dataset)} samples via --limit")

    # Process samples in stable dataset order, in batches. The custom collator
    # keeps each batch as a list of sample dictionaries instead of trying to
    # stack variable-length caption lists; image tensors are prepared below.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda batch: batch,
    )
    print(f"Dataset: {len(dataset)} samples | {len(loader)} batches of {batch_size}")

    # Accumulate one row per sample for each modality, plus the matching labels
    # and IDs. DataLoader shuffling is disabled, so all four outputs retain
    # their shared sample order across batches.
    all_image_embeddings, all_text_embeddings = [], []
    all_labels, all_image_ids = [], []

    t0 = time.time()
    for batch_idx, batch in enumerate(tqdm(loader, desc="extract")):
        # Unpack the per-image records. Captions remain grouped by image here;
        # labels are stacked into a (batch, classes) tensor.
        images = [s["image"] for s in batch]
        captions_per_img = [s["captions"] for s in batch]
        img_ids = [s["image_id"] for s in batch]
        labels = torch.stack([s["labels"] for s in batch], dim=0)

        # ---- Image features ----
        # Raw images are PIL RGB objects, not model inputs. CLIP's image
        # processor resizes/crops and normalizes them into batched tensors;
        # move every produced tensor to the selected inference device.
        img_inputs = processor.image_processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        # Encode images with CLIP's vision encoder. This returns numerical
        # representations, not pixels; projection handling below ensures the
        # standard model's 512-dimensional image embedding is used.
        img_raw = model.get_image_features(**img_inputs)

        if batch_idx == 0:
            print(f"\n[diag] get_image_features returns: {type(img_raw).__name__}")
            if not isinstance(img_raw, torch.Tensor):
                print(f"[diag] attrs: "
                      f"{[a for a in dir(img_raw) if not a.startswith('_')]}")

        img_feats = _as_feature_tensor(
            img_raw,
            projection=getattr(model, "visual_projection", None),
            expected_dim=512,
            tag="image",
        )
        # Normalize each image embedding independently to unit L2 length so
        # that feature magnitude does not dominate when modalities are fused.
        img_feats = _l2norm(img_feats)

        # ---- Text features: flatten captions, encode, mean per image ----
        # Tokenize all captions in this batch together. Keep a parallel owner
        # index for every flattened caption so the encoded caption vectors can
        # later be regrouped with the image and label they describe.
        flat_caps, cap_to_img = [], []
        for i, caps in enumerate(captions_per_img):
            for c in caps:
                flat_caps.append(c)
                cap_to_img.append(i)

        # Tokenization converts raw caption strings to CLIP's token IDs and
        # attention masks. Padding makes the variable-length captions a batch;
        # truncation enforces the tokenizer's maximum sequence length.
        txt_inputs = processor.tokenizer(
            flat_caps,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
        # Encode tokenized captions with CLIP's text encoder, producing one
        # projected vector per caption (rather than one vector per image yet).
        txt_raw = model.get_text_features(**txt_inputs)

        if batch_idx == 0:
            print(f"[diag] get_text_features returns: {type(txt_raw).__name__}")

        txt_feats = _as_feature_tensor(
            txt_raw,
            projection=getattr(model, "text_projection", None),
            expected_dim=512,
            tag="text",
        )
        # Normalize each caption vector before combining multiple captions for
        # one image, giving captions comparable vector scale.
        txt_feats = _l2norm(txt_feats)

        # Sum normalized caption vectors by their owning image, and count the
        # captions in each group. Dividing by the count makes one text vector
        # per image; the final normalization restores unit length after
        # averaging. This associates text with images through cap_to_img, not
        # through position in the original captions JSON.
        cap_to_img_t = torch.tensor(cap_to_img, device=device)
        B = len(batch)
        summed = torch.zeros(B, txt_feats.size(-1), device=device)
        counts = torch.zeros(B, device=device)
        summed.index_add_(0, cap_to_img_t, txt_feats)
        counts.index_add_(
            0, cap_to_img_t, torch.ones_like(cap_to_img_t, dtype=torch.float)
        )
        mean_txt = summed / counts.clamp_min(1e-8).unsqueeze(-1)
        mean_txt = _l2norm(mean_txt)

        # Keep encoder work on the selected device, but move embeddings to CPU
        # before accumulating them for serialization. Labels and IDs are
        # appended in the same batch order to preserve row-wise association.
        all_image_embeddings.append(img_feats.cpu())
        all_text_embeddings.append(mean_txt.cpu())
        all_labels.append(labels)
        all_image_ids.extend(img_ids)

    elapsed = time.time() - t0
    print(f"\nExtraction took {elapsed:.1f}s for {len(all_image_ids)} samples")

    if not all_image_embeddings:
        sys.exit("No features were extracted. Check errors above.")

    output_path = os.path.join(output_dir, "extracted_features.pt")
    tmp_path = output_path + ".tmp"

    # Each embedding matrix has shape (number of samples, 512), labels have
    # shape (number of samples, 80), and image_ids is the matching ordered list
    # of COCO IDs. These are numerical representations/targets, not the
    # original pixels or caption strings.
    # Provenance records the model/runtime and extraction settings for
    # traceability. The evaluator loads the two embedding matrices and labels,
    # then splits and trains its fusion classifiers on these cached features.
    payload = {
        "image_embeddings": torch.cat(all_image_embeddings, dim=0),
        "text_embeddings": torch.cat(all_text_embeddings, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "image_ids": all_image_ids,
        "provenance": {
            "model_name": model_name,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "seed": seed,
            "batch_size": batch_size,
            "device": str(device),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "num_samples": len(all_image_ids),
            "extraction_seconds": round(elapsed, 2),
        },
    }

    # Write to a temporary path first so a failed/incomplete write cannot
    # replace a previously valid feature file.
    torch.save(payload, tmp_path)

    # Check that the serialized file is close to the expected uncompressed
    # storage for two 512-float32 embeddings and one 80-float32 label vector
    # per sample. The tolerance allows serialization metadata overhead.
    expected_min_bytes = (
        len(all_image_ids) * 512 * 4 * 2   # image + text embeddings (float32)
        + len(all_image_ids) * 80 * 4       # labels
    )
    on_disk = os.path.getsize(tmp_path)
    if on_disk < expected_min_bytes * 0.9:
        raise RuntimeError(
            f"Saved file too small: {on_disk:,} bytes; expected >= {expected_min_bytes:,}. "
            f"Write failed."
        )
    if on_disk > expected_min_bytes * 1.5:
        raise RuntimeError(
            f"Saved file too large: {on_disk:,} bytes; expected ~{expected_min_bytes:,}. "
            f"Something re-encoded the file (likely a text-mode tool touched it)."
        )

    # PyTorch's saved archive is expected to use a ZIP ("PK") header.
    with open(tmp_path, "rb") as f:
        head = f.read(4)
    if head[:2] != b"PK":
        raise RuntimeError(f"Saved file has bad header: {head!r}")

    # Reload the temporary archive and verify row counts and feature shapes
    # before promoting it to the final output path.
    _check = torch.load(tmp_path, weights_only=False)
    n = len(all_image_ids)
    assert _check["image_embeddings"].shape == torch.Size([n, 512]), \
        f"image_embeddings shape wrong: {_check['image_embeddings'].shape}"
    assert _check["text_embeddings"].shape == torch.Size([n, 512]), \
        f"text_embeddings shape wrong: {_check['text_embeddings'].shape}"
    assert _check["labels"].shape == torch.Size([n, 80]), \
        f"labels shape wrong: {_check['labels'].shape}"
    del _check

    # Replace the destination only after all sanity checks pass. os.replace
    # makes the completed temporary file the visible final artifact in one
    # operation.
    os.replace(tmp_path, output_path)
    print(f"Saved and verified {len(all_image_ids)} samples to {output_path} "
          f"({on_disk:,} bytes)")


if __name__ == "__main__":
    # The script entry point exposes batching and an optional sample limit.
    # Other paths, model choice, and extraction settings use the function
    # defaults unless the function is called directly from Python.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit dataset size (useful for testing)")
    args = parser.parse_args()
    extract_and_save_features(
        batch_size=args.batch_size,
        limit=args.limit,
    )