"""Convert an existing YOLO dataset into the three PPE detection classes."""

from hashlib import sha256
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import yaml

from ppe.assessment import CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALIASES = {
    "person": "person",
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard hat": "helmet",
    "vest": "vest",
    "safety vest": "vest",
}


def class_map(names: dict | list) -> dict[int, int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for index, value in items:
        name = str(value).strip().lower().replace("_", " ").replace("-", " ")
        if name in ALIASES:
            mapping[int(index)] = CLASSES.index(ALIASES[name])
    missing = [name for i, name in enumerate(CLASSES) if i not in mapping.values()]
    if missing:
        raise ValueError(f"Dataset lacks required classes: {', '.join(missing)}")
    return mapping


def _images(directory: Path) -> list[Path]:
    images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not images:
        raise ValueError(f"No images found in {directory}")
    return images


def _convert_labels(path: Path, mapping: dict[int, int]) -> str:
    if not path.is_file():
        return ""
    output = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            index, *coordinates = line.split()
            target = mapping.get(int(index))
            x, y, width, height = map(float, coordinates)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("coordinates out of range")
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label at {path}:{number}: {line}") from error
        if target is not None:
            output.append(" ".join([str(target), *coordinates]) + "\n")
    return "".join(output)


def prepare_dataset(config_path: Path, destination: Path) -> Path:
    config_path = config_path.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    mapping = class_map(config["names"])
    # Ultralytics archives can declare a root relative to their original download
    # directory. Prefer image paths next to the extracted YAML when they exist.
    config_dir = config_path.parent
    declared_root = config_dir / config.get("path", ".")

    def split_dir(split: str) -> Path:
        path = Path(config[split])
        if not (config_dir / path).is_dir():
            return (declared_root / path).resolve()
        return (config_dir / path).resolve()

    splits = {}
    for name in ("train", "val", "test"):
        if config.get(name):
            directory = split_dir(name)
            if "images" not in directory.parts:
                raise ValueError(f"Expected a YOLO images directory for {name}: {directory}")
            splits[name] = [(directory, image) for image in _images(directory)]
    if "test" not in splits:
        # Use groups of byte-identical files so they stay on the same side.
        groups = {}
        for directory, image in splits["val"]:
            digest = sha256(image.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append((directory, image))
        if len(groups) < 2:
            raise ValueError("At least two distinct validation images are required to create a test split")
        ordered = sorted(groups)
        midpoint = len(ordered) // 2
        splits["test"] = [item for digest in ordered[:midpoint] for item in groups[digest]]
        splits["val"] = [item for digest in ordered[midpoint:] for item in groups[digest]]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "dataset"
        for split, entries in splits.items():
            for directory, image in entries:
                relative = image.relative_to(directory)
                # Replace the closest directory named images with labels.
                parts = list(directory.parts)
                index = len(parts) - 1 - parts[::-1].index("images")
                parts[index] = "labels"
                label_relative = relative.with_suffix(".txt")
                label = Path(*parts) / label_relative
                output_image = staging / "images" / split / relative
                output_label = staging / "labels" / split / label_relative
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                copy2(image, output_image)
                output_label.write_text(_convert_labels(label, mapping), encoding="utf-8")
        output_config = {
            "path": str(destination),
            **{split: f"images/{split}" for split in ("train", "val", "test")},
            "names": list(CLASSES),
        }
        (staging / "data.yaml").write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")
        staging.rename(destination)
    return destination / "data.yaml"
