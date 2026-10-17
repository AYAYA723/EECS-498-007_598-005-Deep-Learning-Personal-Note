"""Local/Colab runtime and public course assets for the A6 notebooks."""

import os
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

import torch
import torchvision


PROJECT_PATH = Path(__file__).resolve().parent
DATA_DIR = PROJECT_PATH / "datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 2


def mnist_root():
    """Reuse complete local MNIST files before requesting another download."""
    candidates = [
        DATA_DIR / "MNIST_data",
        PROJECT_PATH / "MNIST_data",
        PROJECT_PATH.parent / "assignment2" / "mnist_dataset",
    ]
    raw_names = (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )
    for root in candidates:
        if all((root / "MNIST" / "raw" / name).is_file() for name in raw_names):
            return root
    return candidates[0]


def download_file(urls, destination):
    """Download to a temporary file; never leave an incomplete cached asset."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    errors = []
    for url in urls:
        try:
            print(f"Downloading {destination.name} ...")
            with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if not temporary.stat().st_size:
                raise OSError("The server returned an empty file.")
            temporary.replace(destination)
            return destination
        except (OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            errors.append(f"{url}: {error}")
    raise RuntimeError(
        f"Could not download {destination.name}. Check the network or place the "
        f"course file at {destination}.\n" + "\n".join(errors)
    )


def ensure_imagenet_val():
    return download_file(
        ["https://cs231n.stanford.edu/imagenet_val_25.npz"],
        DATA_DIR / "imagenet_val_25.npz",
    )


def ensure_style_data():
    """Extract only the course images and reference arrays, without shell tools."""
    target = DATA_DIR / "style_transfer"
    required = [target / "style-transfer-checks.npz"] + [
        target / "styles" / name for name in
        ("tubingen.jpg", "starry_night.jpg", "composition_vii.jpg", "the_scream.jpg")
    ]
    if all(path.is_file() for path in required):
        return target
    archive = download_file(
        # The old UMich style_data.zip URL is no longer available. These are
        # the original CS231n SqueezeNet style images and reference arrays.
        ["https://cs231n.stanford.edu/assignments/2017/spring1617_assignment3_v3.zip"],
        DATA_DIR / "style_data.zip",
    )
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = Path(member.filename).parts
            if member.is_dir() or ".." in parts:
                continue
            if "styles" in parts:
                relative = Path("styles") / parts[-1]
            elif parts[-1] == "style-transfer-checks.npz":
                relative = Path(parts[-1])
            else:
                continue
            destination = target / relative
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete style-transfer assets: " + ", ".join(missing))
    return target


def load_squeezenet():
    """Load the original ImageNet weights into a project-local, hash-checked cache."""
    weights = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
    state = torch.hub.load_state_dict_from_url(
        weights.url,
        model_dir=str(DATA_DIR / "weights"),
        map_location="cpu",
        check_hash=True,
        weights_only=True,
    )
    model = torchvision.models.squeezenet1_1(weights=None)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()
