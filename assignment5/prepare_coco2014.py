"""Build a reproducible A5-compatible subset from original COCO 2014 data.

This is NOT the unavailable Winter 2022 course coco.pt. No homework solutions
are implemented here. Images remain RGB uint8; ImageEncoder normalizes them.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch


SOURCE_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATIONS_URL = SOURCE_BASE + "/annotations/annotations_trainval2014.zip"
SPECIAL_TOKENS = ["<NULL>", "<START>", "<END>", "<UNK>"]
ANNOTATIONS_MD5 = "0a379cfc70b0e71301e0f377548639bd"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, retries=4):
    """Reuse completed files, resume partial transfers, then rename atomically."""
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "EECS498-A5-COCO2014-rebuild/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                resume = offset > 0 and response.status == 206
                initial_size = offset if resume else 0
                expected = response.headers.get("Content-Length")
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise OSError("Unexpected download range")
                with partial.open("ab" if resume else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if expected and partial.stat().st_size != initial_size + int(expected):
                    raise OSError("Incomplete download")
            partial.replace(destination)
            return destination
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def extract_captions(raw_dir):
    archive = raw_dir / "annotations_trainval2014.zip"
    files = [raw_dir / "annotations" / f"captions_{split}2014.json"
             for split in ("train", "val")]
    if all(path.is_file() for path in files):
        return files
    print("Downloading/verifying official COCO 2014 annotations...", flush=True)
    partial = archive.with_name(archive.name + ".part")
    # An external curl transfer may already have completed the partial file.
    if not archive.exists() and partial.exists() and file_hash(partial, "md5") == ANNOTATIONS_MD5:
        partial.replace(archive)
    download(ANNOTATIONS_URL, archive)
    if file_hash(archive, "md5") != ANNOTATIONS_MD5:
        raise ValueError(f"Annotation checksum mismatch: {archive}. Keep it for inspection and download a fresh copy.")
    with zipfile.ZipFile(archive) as zipped:
        for path in files:
            member = "annotations/" + path.name
            # Extract exactly these two known members, never arbitrary archive paths.
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.part")
            with zipped.open(member) as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            temporary.replace(path)
    return files


def tokenize(caption):
    # Lowercase English words/numbers; keep internal apostrophes, remove punctuation.
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", caption.lower())


def select_pairs(annotation_path, count, seed, captions_per_image):
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for annotation in annotations["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    images = {image["id"]: image for image in annotations["images"]}
    eligible = sorted(image_id for image_id in images
                      if len(by_image[image_id]) >= captions_per_image)
    image_count = (count + captions_per_image - 1) // captions_per_image
    if image_count > len(eligible):
        raise ValueError("Requested more COCO images than are available")
    rng = random.Random(seed)
    selected = rng.sample(eligible, image_count)
    pairs = []
    for image_id in selected:
        captions = sorted(by_image[image_id], key=lambda annotation: annotation["id"])
        for annotation in rng.sample(captions, captions_per_image):
            pairs.append({"image_id": image_id, "annotation_id": annotation["id"],
                          "file_name": images[image_id]["file_name"],
                          "caption": annotation["caption"]})
    pairs = pairs[:count]
    rng.shuffle(pairs)
    return pairs


def build_vocab(train_pairs, min_frequency, max_words):
    counts = Counter(word for pair in train_pairs
                     for word in tokenize(pair["caption"])[:max_words])
    words = sorted(word for word, count in counts.items() if count >= min_frequency)
    idx_to_token = SPECIAL_TOKENS + words
    return {"idx_to_token": idx_to_token,
            "token_to_idx": {token: index for index, token in enumerate(idx_to_token)}}


def encode_captions(pairs, vocab, max_words):
    mapping = vocab["token_to_idx"]
    # 15 content words + START + END = 17 tokens by default.
    captions = torch.full((len(pairs), max_words + 2), mapping["<NULL>"], dtype=torch.int64)
    for i, pair in enumerate(pairs):
        words = tokenize(pair["caption"])[:max_words]
        tokens = [mapping["<START>"]]
        tokens.extend(mapping.get(word, mapping["<UNK>"]) for word in words)
        tokens.append(mapping["<END>"])
        captions[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int64)
    return captions


def download_images(pairs, split, raw_dir, workers):
    names = sorted({pair["file_name"] for pair in pairs})
    folder = raw_dir / f"{split}2014"

    def get_image(name):
        if Path(name).name != name or not name.endswith(".jpg"):
            raise ValueError(f"Invalid COCO image filename: {name}")
        path = download(f"{SOURCE_BASE}/{split}2014/{name}", folder / name)
        with Image.open(path) as image:
            image.verify()
        return path

    print(f"Downloading/reusing {len(names)} original {split} images ({workers} workers)...", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(get_image, name): name for name in names}
        for completed, job in enumerate(as_completed(jobs), 1):
            try:
                job.result()
            except Exception as error:
                failures.append((jobs[job], str(error)))
            if completed % 100 == 0 or completed == len(names):
                print(f"  {split}: {completed}/{len(names)} checked; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} image downloads failed. Rerun to reuse completed files. Examples: {failures[:3]}")


def make_images(pairs, split, raw_dir, image_size):
    images = torch.empty((len(pairs), 3, image_size, image_size), dtype=torch.uint8)
    # Cache only resized images: repeated captions share exactly the same pixels.
    resized = {}
    for i, pair in enumerate(pairs):
        name = pair["file_name"]
        if name not in resized:
            with Image.open(raw_dir / f"{split}2014" / name) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
                resized[name] = torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        images[i].copy_(resized[name])
    return images


def validate_data(data, manifest, train_count, val_count, image_size, max_words):
    mapping = data["vocab"]["token_to_idx"]
    assert len(mapping) == len(data["vocab"]["idx_to_token"])
    assert all(mapping[word] == i for i, word in enumerate(data["vocab"]["idx_to_token"]))
    train_ids = {pair["image_id"] for pair in manifest["train"]}
    val_ids = {pair["image_id"] for pair in manifest["val"]}
    assert not train_ids.intersection(val_ids), "Image leakage between train and validation"
    for split, count in (("train", train_count), ("val", val_count)):
        images, captions = data[f"{split}_images"], data[f"{split}_captions"]
        assert images.shape == (count, 3, image_size, image_size)
        assert images.dtype == torch.uint8 and images.device.type == "cpu"
        assert captions.shape == (count, max_words + 2)
        assert captions.dtype == torch.int64 and captions.device.type == "cpu"
        assert int(captions.min()) >= 0 and int(captions.max()) < len(mapping)
        assert (captions[:, 0] == mapping["<START>"]).all()
        assert ((captions == mapping["<END>"]).sum(dim=1) == 1).all()
        for row in captions:
            end = int((row == mapping["<END>"]).nonzero()[0, 0])
            assert (row[end + 1:] == mapping["<NULL>"]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "datasets")
    parser.add_argument("--train-pairs", type=int, default=10000)
    parser.add_argument("--val-pairs", type=int, default=500)
    parser.add_argument("--captions-per-image", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-words", type=int, default=15)
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if min(args.train_pairs, args.val_pairs, args.captions_per_image, args.image_size,
           args.max_words, args.min_frequency, args.workers) < 1:
        parser.error("Counts, sizes, frequencies, and workers must be positive")
    args.data_dir = args.data_dir.expanduser().resolve()
    output = args.data_dir / "coco.pt"
    if output.exists():
        parser.error(f"Refusing to overwrite existing data: {output}")
    raw_dir = args.data_dir / "coco2014_raw"
    train_annotations, val_annotations = extract_captions(raw_dir)
    train_pairs = select_pairs(train_annotations, args.train_pairs, args.seed, args.captions_per_image)
    val_pairs = select_pairs(val_annotations, args.val_pairs, args.seed + 1, args.captions_per_image)
    vocab = build_vocab(train_pairs, args.min_frequency, args.max_words)
    print(f"Vocabulary: {len(vocab['idx_to_token'])} tokens (train only)", flush=True)
    manifest = {"train": train_pairs, "val": val_pairs}
    for split, pairs in manifest.items():
        download_images(pairs, split, raw_dir, args.workers)
    data = {"train_images": make_images(train_pairs, "train", raw_dir, args.image_size),
            "val_images": make_images(val_pairs, "val", raw_dir, args.image_size),
            "train_captions": encode_captions(train_pairs, vocab, args.max_words),
            "val_captions": encode_captions(val_pairs, vocab, args.max_words), "vocab": vocab}
    validate_data(data, manifest, args.train_pairs, args.val_pairs, args.image_size, args.max_words)
    partial = output.with_suffix(".pt.part")
    torch.save(data, partial)
    reloaded = torch.load(partial, map_location="cpu", weights_only=True)
    validate_data(reloaded, manifest, args.train_pairs, args.val_pairs, args.image_size, args.max_words)
    partial.replace(output)
    metadata = {"original_course_file": False, "source": "MS COCO 2014 official train/val splits",
                "source_base": SOURCE_BASE, "annotations_url": ANNOTATIONS_URL,
                "terms": "https://cocodataset.org/#termsofuse",
                "annotations_md5": ANNOTATIONS_MD5,
                "caption_annotations_sha256": {p.name: file_hash(p) for p in (train_annotations, val_annotations)},
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "tokenization": "lowercase; words/numbers with internal apostrophes; punctuation removed",
                "caption_length": args.max_words + 2,
                "vocab_size": len(vocab["idx_to_token"]),
                "image_storage": "RGB CHW uint8 0..255; square bilinear resize, no normalization",
                "coco_pt_sha256": file_hash(output),
                "unique_images": {split: len({pair['image_id'] for pair in pairs}) for split, pairs in manifest.items()},
                "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    for name, content in (("coco2014_manifest.json", manifest), ("coco2014_metadata.json", metadata)):
        path = args.data_dir / name
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    print(f"Saved and verified: {output} ({output.stat().st_size:,} bytes)", flush=True)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)} {value.dtype}", flush=True)


if __name__ == "__main__":
    main()
