# A5 COCO 2014 replacement dataset

`datasets/coco.pt` is rebuilt from the original MS COCO 2014 images and caption
annotations. It is **not** the unavailable Winter 2022 course-provided subset.
It has the same dictionary layout required by `a5_helper.load_coco_captions`.
Samples, word indices, vocabulary size, and training losses differ from the
original course dataset; do not compare those results as if the data were equal.

## Reproduce

Use the same Python environment as the notebook (torch, numpy, and Pillow):

```powershell
& '..\.venv\Scripts\python.exe' -B -u '.\prepare_coco2014.py'
```

Run this from `assignment5`. The notebook's data-loading cell also runs this
script if its default dataset is missing. Downloaded images are cached, and
interrupted downloads can be resumed. An existing `coco.pt` is never overwritten.
To experiment with another subset, use `--data-dir` to choose a new directory.

Defaults:

- Random seed 0; official training and validation splits stay separate.
- 2,000 training images × 5 descriptions = 10,000 image-caption pairs.
- 100 validation images × 5 descriptions = 500 image-caption pairs.
- Select image IDs from a sorted list; select five genuine captions per image;
  shuffle pairs deterministically. No image occurs in both splits.
- RGB, bilinear square resize to 112×112, stored as CHW uint8 in [0, 255].
  Do **not** pre-normalize: the provided `ImageEncoder` handles normalization.
- Lowercase English words/numbers; remove punctuation except internal
  apostrophes. Keep at most 15 content words per caption.
- Build vocabulary only from the selected training captions after truncation.
  Words occurring fewer than 5 times map to `<UNK>`, including unseen validation
  words. Vocabulary order is the four special tokens followed by sorted words.
- Add `<START>` and `<END>`; pad the remaining positions with `<NULL>`.
  Fifteen content words plus the two boundary tokens give a total length of 17.

The tensor layout is:

```text
train_images:   (10000, 3, 112, 112), torch.uint8
val_images:    (500,   3, 112, 112), torch.uint8
train_captions: (10000, 17),         torch.int64
val_captions:   (500,   17),         torch.int64
vocab:
  idx_to_token: list[str]
  token_to_idx: dict[str, int]
```

## Files and provenance

- `datasets/coco2014_raw/annotations_trainval2014.zip`: original annotations.
- `datasets/coco2014_raw/annotations/captions_{train,val}2014.json`: original JSON.
- `datasets/coco2014_raw/{train,val}2014/`: selected original JPEG images only.
- `datasets/coco2014_manifest.json`: exact image IDs, annotation IDs, and captions
  in serialized pair order.
- `datasets/coco2014_metadata.json`: preprocessing settings, source URLs,
  versions, annotation checksums, and the final dataset's SHA-256 hash.

The annotation archive is checked against the official S3 ETag/MD5
`0a379cfc70b0e71301e0f377548639bd`. Every downloaded JPEG is decoded/verified;
the serialized dataset is safely reloaded and validated before completion.
Only the needed images are downloaded, not the roughly 19 GB full image ZIPs.
All data stays local under the repository's existing `datasets/` ignore rule.

Sources: [COCO downloads](https://cocodataset.org/#download),
[official caption API](https://github.com/cocodataset/cocoapi),
[official S3 annotation archive](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip).

COCO annotation data is licensed under CC BY 4.0. Individual image rights
remain with the respective creators; see the
[COCO Terms of Use](https://cocodataset.org/#termsofuse). No dataset is uploaded
to GitHub by this script.
