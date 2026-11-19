from pathlib import Path

import pytest
import yaml

from ppe.dataset import prepare_dataset


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True)
        (root / "images" / split / "a.jpg").write_bytes(split.encode())
        (root / "labels" / split / "a.txt").write_text(
            "0 0.1 0.2 0.1 0.1\n5 0.2 0.3 0.4 0.5\n"
            "2 0.5 0.5 0.3 0.3\n11 0.5 0.5 0.3 0.3\n",
        )
    config = root / "data.yaml"
    config.write_text(
        "path: archive-name\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names: {0: helmet, 5: Person, 2: vest, 11: no_vest}\n",
    )
    return config


def test_prepare_preserves_splits_and_remaps_classes(source, tmp_path):
    destination = tmp_path / "converted"
    config = yaml.safe_load(prepare_dataset(source, destination).read_text())
    assert config["names"] == ["person", "helmet", "vest"]
    assert Path(config["path"]) == destination
    for split in ("train", "val", "test"):
        assert (destination / config[split] / "a.jpg").read_bytes() == split.encode()
        assert (destination / "labels" / split / "a.txt").read_text().splitlines() == [
            "1 0.1 0.2 0.1 0.1", "0 0.2 0.3 0.4 0.5", "2 0.5 0.5 0.3 0.3",
        ]


def test_prepare_creates_deterministic_test_split(source, tmp_path):
    source.write_text(source.read_text().replace("test: images/test\n", ""))
    val = source.parent / "images/val"
    (val / "b.jpg").write_bytes(b"another example")
    (val / "copy.jpg").write_bytes(b"val")
    splits = []
    for directory in ("first", "second"):
        root = prepare_dataset(source, tmp_path / directory).parent
        splits.append([{p.name for p in (root / "images" / kind).iterdir()}
                       for kind in ("val", "test")])
    assert splits[0] == splits[1]
    val_names, test_names = splits[0]
    assert not (val_names & test_names)
    assert val_names | test_names == {"a.jpg", "b.jpg", "copy.jpg"}
    assert ("a.jpg" in val_names) == ("copy.jpg" in val_names)


def test_invalid_annotation_leaves_no_partial_dataset(source, tmp_path):
    (source.parent / "labels/test/a.txt").write_text("2 0.5 0.5 0.2")
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="Invalid YOLO label"):
        prepare_dataset(source, destination)
    assert not destination.exists()
