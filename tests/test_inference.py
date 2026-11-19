from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("ultralytics")

from ppe import inference
from ppe.assessment import Assessment, Box, Status


@pytest.mark.parametrize(("helmet", "vest", "caption"), [
    (Status.PRESENT, Status.PRESENT, "СИЗ в порядке"),
    (Status.MISSING, Status.PRESENT, "Без каски"),
    (Status.PRESENT, Status.MISSING, "Без жилета"),
    (Status.MISSING, Status.MISSING, "Без каски и жилета"),
    (Status.UNKNOWN, Status.MISSING, "Каска под вопросом, без жилета"),
    (Status.UNKNOWN, Status.UNKNOWN, "СИЗ не удалось проверить"),
])
def test_russian_captions(helmet, vest, caption):
    assert inference._caption(Assessment(Box(0, 0, 1, 1), helmet, vest)) == caption


def test_annotates_photo_and_video(tmp_path, monkeypatch):
    box = np.array([[15, 10, 80, 90]])
    names = {0: "person", 1: "helmet", 2: "vest"}
    detection = SimpleNamespace(names=names, boxes=SimpleNamespace(cls=np.array([0]), xyxy=box))
    keypoints = np.zeros((1, 17, 3))
    for i, x, y in ((0, 45, 20), (1, 43, 18), (5, 30, 30), (6, 60, 30),
                    (11, 35, 55), (12, 55, 55)):
        keypoints[0, i] = [x, y, 0.9]
    pose = SimpleNamespace(boxes=SimpleNamespace(xyxy=box), keypoints=SimpleNamespace(data=keypoints))

    def model(path):
        result = pose if "pose" in path else detection
        return SimpleNamespace(names=names, task="detect", predict=lambda *_a, **_kw: [result])

    monkeypatch.setattr(inference, "YOLO", model)
    monitor = inference.PPEMonitor(tmp_path / "ppe.pt", "cpu", 0.35, 320)
    frame = np.zeros((96, 320, 3), dtype=np.uint8)
    original = tmp_path / "кадр.png"
    cv2.imencode(".png", frame)[1].tofile(original)
    output = tmp_path / "результат.png"
    inference.process(monitor, original, output)
    rendered = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert tuple(rendered[90, 15]) == (30, 30, 220)

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 96))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(frame)
    writer.release()
    video = tmp_path / "result.mp4"
    inference.process(monitor, source, video)
    reader = cv2.VideoCapture(str(video))
    try:
        assert reader.get(cv2.CAP_PROP_FPS) == 10
        for _ in range(3):
            ok, rendered = reader.read()
            assert ok and rendered[:, :, 2].max() > 140
        assert not reader.read()[0]
    finally:
        reader.release()


def test_empty_image_does_not_replace_output(tmp_path):
    source = tmp_path / "empty.jpg"
    source.touch()
    output = tmp_path / "existing.jpg"
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="Cannot decode image"):
        inference.process(None, source, output)
    assert output.read_bytes() == b"existing output"
