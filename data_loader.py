import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class COCOMultiModalDataset(Dataset):
    """COCO image-caption samples with multi-label targets from instances."""

    # This class implements PyTorch's Dataset indexing interface; it does not
    # create a DataLoader, so batching and shuffling are configured by callers.
    def __init__(self, image_dir, caption_json, instance_json, transform=None):
        # Keep the image root and optional preprocessing callable for use in
        # __getitem__. The two JSON files are supplied by the caller, so they
        # determine which COCO annotation split this dataset represents.
        self.image_dir = image_dir
        self.transform = transform

        # Read the COCO captions file. Each annotation associates one caption
        # string with an image_id; grouping by that ID preserves all captions
        # belonging to an image rather than selecting only one of them.
        with open(caption_json, "r", encoding="utf-8") as f:
            cap_data = json.load(f)

        self.img_id_to_captions = {}
        for ann in cap_data["annotations"]:
            self.img_id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])

        # Read the COCO instances file, which supplies the image records,
        # category definitions, and object annotations used to make labels.
        with open(instance_json, "r", encoding="utf-8") as f:
            inst_data = json.load(f)

        # Fix the label-vector order by sorting categories by COCO category ID.
        # The map then translates the category_id stored in each annotation
        # into the corresponding position in that vector.
        self.categories = sorted(inst_data["categories"], key=lambda x: x["id"])
        self.cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(self.categories)}

        # Instance metadata is the source of the image ID-to-filename mapping.
        # Allocate one zero-filled, float32 multi-hot vector per listed image;
        # an image with no recognized object annotations therefore keeps an
        # all-zero vector. The vector length is the number of categories in
        # this instance file (80 for the standard COCO detection annotations).
        img_id_to_filename = {img["id"]: img["file_name"] for img in inst_data["images"]}
        self.img_id_to_labels = {
            img["id"]: np.zeros(len(self.categories), dtype=np.float32)
            for img in inst_data["images"]
        }
        # Every instance annotation marks its category's position as present.
        # Repeated objects in one category do not increase the value: this is
        # multi-label presence, not an object-count target.
        for ann in inst_data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id in self.cat_id_to_idx:
                self.img_id_to_labels[ann["image_id"]][self.cat_id_to_idx[cat_id]] = 1.0

        # Keep instance-file image IDs only when a caption group exists. This
        # creates the initial candidate set, independently of whether the
        # image file is present on disk.
        candidates = sorted(
            img_id for img_id in img_id_to_filename
            if img_id in self.img_id_to_captions
        )
        # A usable sample must also have an existing file at the path formed
        # from image_dir and the COCO filename. Missing files are counted and
        # omitted; no other filtering (such as label-content filtering) occurs.
        kept, dropped_missing_file = [], 0
        for img_id in candidates:
            path = os.path.join(image_dir, img_id_to_filename[img_id])
            if os.path.exists(path):
                kept.append(img_id)
            else:
                dropped_missing_file += 1

        # This ordered ID list is the dataset index: each retained ID can now
        # be used to retrieve its filename, caption list, and label vector.
        self.valid_image_ids = kept
        self.img_id_to_filename = img_id_to_filename

        # Report how many instance-file images had captions and how many
        # candidates survived the on-disk file check.
        n_total = len(img_id_to_filename)
        n_with_caps = len(candidates)
        print(
            f"[COCOMultiModalDataset] total={n_total} | "
            f"with-captions={n_with_caps} | "
            f"missing-file-dropped={dropped_missing_file} | "
            f"kept={len(self.valid_image_ids)}"
        )

    def __len__(self):
        # PyTorch uses this count to determine the number of indexed samples
        # available from this Dataset.
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        # Resolve one dataset index to the shared COCO image ID. That ID is the
        # join key linking the instance image record to its captions and labels.
        img_id = self.valid_image_ids[idx]
        img_path = os.path.join(self.image_dir, self.img_id_to_filename[img_id])

        # Load the image while the file is open, and standardize its color
        # mode to RGB (three channels) before optional caller-supplied
        # preprocessing. The transform's output is returned as-is.
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)

        # A sample contains the COCO ID, the RGB image (or transformed image),
        # every caption grouped for that ID as a Python list of strings, and a
        # float32 PyTorch tensor whose length is the number of sorted
        # categories. Captions and labels are paired to this image by ID.
        # Split selection is not implemented here: the JSON paths and image
        # directory passed to the constructor determine the data being read.
        return {
            "image_id": img_id,
            "image": image,
            "captions": self.img_id_to_captions[img_id],
            "labels": torch.tensor(self.img_id_to_labels[img_id], dtype=torch.float32),
        }