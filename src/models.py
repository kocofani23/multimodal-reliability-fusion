import torch
import torch.nn as nn

# These modules operate on feature vectors and do not encode or normalize raw
# modalities. Missingness is handled by explicit masks; there are no internal
# stochastic modality-dropout, dropout, batch-normalization, or layer-
# normalization layers in the architectures below.


def _to_mask_tensor(mask, batch_size, device):
    """bool -> per-batch flag; (B,) tensor -> per-sample flags."""
    if isinstance(mask, bool):
        # A Python bool applies the same availability choice to every item.
        return torch.full((batch_size,), mask, dtype=torch.bool, device=device)
    if isinstance(mask, torch.Tensor):
        # Tensor masks allow different items in the batch to lose different
        # modalities. True means masked/unavailable; callers pass either a
        # shared bool or one flag per batch item.
        return mask.to(device).bool()
    raise TypeError(f"Unsupported mask type: {type(mask)}")


class MultiLabelClassifier(nn.Module):
    """Map a feature vector to one logit per target category."""

    def __init__(self, input_dim, num_classes=80, hidden_dim=256):
        super().__init__()
        # The first linear layer mixes input features into a smaller hidden
        # representation; ReLU adds a non-linear decision boundary. The last
        # layer emits independent class logits (80 by default). No sigmoid is
        # applied here; training/loss or inference code handles logits.
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x is expected to be (B, input_dim); output is (B, num_classes).
        return self.classifier(x)



class LateFusionModel(nn.Module):
    """Baseline: mask each modality, concatenate, and classify jointly."""

    def __init__(self, num_classes=80, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        # With one feature_dim vector per modality, concatenation produces
        # 2 * feature_dim inputs (1024 when the CLIP features are 512-d).
        self.classifier = MultiLabelClassifier(feature_dim * 2, num_classes)

    def forward(self, img_feat, text_feat, mask_image=False, mask_text=False,
                oracle_weights=None):
        # Inputs are paired batches of frozen image/text features, normally
        # shaped (B, feature_dim). oracle_weights is accepted for a uniform
        # model API, but this baseline does not use reliability weights.
        B = img_feat.size(0)
        device = img_feat.device

        # Convert True=unavailable masks into 0/1 availability multipliers.
        # Expanding to (B, 1) lets each flag zero every component of that
        # sample's feature vector. This is explicit mask handling; this class
        # does not randomly drop modalities internally.
        img_avail = (~_to_mask_tensor(mask_image, B, device)).float().unsqueeze(-1)
        txt_avail = (~_to_mask_tensor(mask_text, B, device)).float().unsqueeze(-1)

        img_feat = img_feat * img_avail
        text_feat = text_feat * txt_avail
        # Late fusion keeps both feature vectors as separate halves of one
        # 2*feature_dim representation; the classifier learns how to use them
        # jointly after concatenation.
        z = torch.cat([img_feat, text_feat], dim=-1)
        return self.classifier(z), {}


class GatedConcatFusionModel(nn.Module):
    """
    Reliability-aware fusion that gates each modality before concatenation.

    Unlike late fusion's plain concatenation, a learned (or supplied oracle)
    weight scales each modality before the two scaled vectors are concatenated.
    Missing modalities are indicated by explicit masks. Corruption is
    represented by altered feature values (for example, noise added upstream);
    this model does not itself corrupt features or randomly drop modalities.
    """

    def __init__(self, num_classes=80, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # Gate input contains both feature vectors and two availability flags:
        # (2*feature_dim + 2). The hidden layer/ReLU learns a joint reliability
        # representation, and the final two-way softmax produces nonnegative
        # image/text weights that sum to one per sample.
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )
        # The main classifier receives the gated concatenation, preserving
        # separate image and text sub-vectors (2*feature_dim total).
        self.classifier = MultiLabelClassifier(feature_dim * 2, num_classes)

        # Optional auxiliary classifiers each map one gated modality vector to
        # class logits. They run only when return_aux=True; the main logits are
        # produced independently by classifier above.
        self.aux_img = nn.Linear(feature_dim, num_classes)
        self.aux_txt = nn.Linear(feature_dim, num_classes)

    def forward(self, img_feat, text_feat, mask_image=False, mask_text=False,
                oracle_weights=None, return_aux=False):
        # Inputs are paired (B, feature_dim) feature vectors (typically
        # precomputed frozen CLIP representations), not raw images or strings.
        # This file defines fusion/classification layers, not CLIP encoders.
        B = img_feat.size(0)
        device = img_feat.device

        # A true mask marks a modality as unavailable. Zero its feature and
        # retain a 0/1 availability indicator for the gate to observe;
        # corrupted-but-present features are supplied as values, not masks.
        img_avail = (~_to_mask_tensor(mask_image, B, device)).float().unsqueeze(-1)
        txt_avail = (~_to_mask_tensor(mask_text, B, device)).float().unsqueeze(-1)

        img_clean = img_feat * img_avail
        txt_clean = text_feat * txt_avail

        if oracle_weights is not None:
            # Supplied two-column weights bypass the learned gate; each column
            # is the image/text multiplier for the corresponding batch item.
            cw = oracle_weights.to(device).float()
            w_i, w_t = cw[:, 0:1], cw[:, 1:2]
        else:
            # Availability flags are appended to the cleaned features so the
            # gate can distinguish a zero-valued input from an absent one.
            gate_in = torch.cat([img_clean, txt_clean, img_avail, txt_avail], dim=-1)
            w = self.gate_net(gate_in)
            w_i, w_t = w[:, 0:1], w[:, 1:2]

        # Apply one scalar weight to every dimension of each modality, then
        # concatenate the weighted vectors. This is gated concatenation, not
        # weighted summation: the two modalities remain distinct in z.
        gated_img = w_i * img_clean
        gated_txt = w_t * txt_clean
        z = torch.cat([gated_img, gated_txt], dim=-1)
        logits = self.classifier(z)

        # Always expose the per-sample gate weights for inspection/evaluation.
        aux = {"w_image": w_i.squeeze(-1), "w_text": w_t.squeeze(-1)}
        if return_aux:
            # Optional modality-specific logits are computed from the same
            # gated vectors used by the fused classifier.
            aux["img_logits"] = self.aux_img(gated_img)
            aux["txt_logits"] = self.aux_txt(gated_txt)
        return logits, aux


class GatedSumFusionModel(nn.Module):
    """
    Optional ablation: gate each modality and add the vectors elementwise.

    Unlike GatedConcatFusionModel, summation combines matching dimensions into
    a single feature_dim vector before classification, so the classifier input
    is 512-d by default rather than 1024-d.
    """

    def __init__(self, num_classes=80, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        # This gate uses the same feature-plus-availability inputs and
        # two-way softmax weighting scheme as gated concatenation.
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )
        # A summed vector has feature_dim components, not twice that size.
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)

    def forward(self, img_feat, text_feat, mask_image=False, mask_text=False,
                oracle_weights=None):
        # Inputs are paired image/text feature batches; no raw modality
        # encoding or random modality dropout is performed in this model.
        B = img_feat.size(0)
        device = img_feat.device

        # Turn explicit missing-modality masks into availability multipliers
        # and zero unavailable feature vectors before calculating gate weights.
        img_avail = (~_to_mask_tensor(mask_image, B, device)).float().unsqueeze(-1)
        txt_avail = (~_to_mask_tensor(mask_text, B, device)).float().unsqueeze(-1)
        img_feat = img_feat * img_avail
        text_feat = text_feat * txt_avail

        if oracle_weights is not None:
            # As in the gated-concatenation model, oracle weights replace
            # learned gate output when explicitly supplied.
            cw = oracle_weights.to(device).float()
            w_i, w_t = cw[:, 0:1], cw[:, 1:2]
        else:
            # Include availability flags so gate decisions can account for a
            # modality being absent rather than merely having small values.
            gate_in = torch.cat([img_feat, text_feat, img_avail, txt_avail], dim=-1)
            w = self.gate_net(gate_in)
            w_i, w_t = w[:, 0:1], w[:, 1:2]

        # Elementwise weighted addition fuses corresponding dimensions:
        # z[j] = w_image * image[j] + w_text * text[j]. The classifier returns
        # one logit per class; gate weights are included for diagnostics.
        z = w_i * img_feat + w_t * text_feat
        return self.classifier(z), {"w_image": w_i.squeeze(-1), "w_text": w_t.squeeze(-1)}