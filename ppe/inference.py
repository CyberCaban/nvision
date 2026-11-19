"""Run YOLOv8 detection and pose estimation on photos and videos."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ppe.assessment import CLASSES, Assessment, Box, Detection, Pose, Status, assess
from ppe.dataset import IMAGE_SUFFIXES

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ITEM_LABELS = {"helmet": "Каска", "vest": "Жилет"}


class PPEMonitor:
    def __init__(self, weights: Path, device: str | None, confidence: float, imgsz: int):
        self.detector = YOLO(str(weights))
        if self.detector.task != "detect" or set(self.detector.names.values()) != set(CLASSES):
            raise ValueError(f"Use PPE detection weights with these classes: {', '.join(CLASSES)}")
        self.pose = YOLO("yolov8n-pose.pt")
        self.options = {"device": device, "conf": confidence, "imgsz": imgsz, "verbose": False}
        font_path = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        self.font = ImageFont.truetype(str(font_path), 14)

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        result = self.detector.predict(frame, **self.options)[0]
        detections = [
            Detection(result.names[int(index)], Box(*coordinates))
            for index, coordinates in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())
        ]
        people = []
        if any(item.label == "person" for item in detections):
            pose_result = self.pose.predict(frame, **self.options)[0]
            if pose_result.keypoints is not None:
                people = [
                    Pose(Box(*coordinates), tuple(map(tuple, keypoints)))
                    for coordinates, keypoints in zip(
                        pose_result.boxes.xyxy.tolist(), pose_result.keypoints.data.tolist(),
                    )
                ]
        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        annotations = []
        for person in assess(detections, people, width, height):
            if Status.MISSING in (person.helmet, person.vest):
                color = (220, 30, 30)
            elif Status.UNKNOWN in (person.helmet, person.vest):
                color = (150, 89, 0)
            else:
                color = (30, 170, 30)
            annotations.append((person.person, _caption(person), color, 2))
        annotations.extend(
            (item.box, ITEM_LABELS[item.label], (20, 109, 160), 1)
            for item in detections if item.label in ITEM_LABELS
        )
        for box, _, color, thickness in annotations:
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=thickness)
        occupied = []
        for box, label, color, _ in annotations:
            _draw_label(draw, image.size, self.font, box, label, color, occupied)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _caption(person: Assessment) -> str:
    helmet, vest = person.helmet, person.vest
    if helmet == vest == Status.PRESENT:
        return "СИЗ в порядке"
    if helmet == vest == Status.MISSING:
        return "Без каски и жилета"
    if helmet == vest == Status.UNKNOWN:
        return "СИЗ не удалось проверить"
    labels = []
    for status, missing, unknown in (
        (helmet, "без каски", "каска под вопросом"),
        (vest, "без жилета", "жилет под вопросом"),
    ):
        if status == Status.MISSING:
            labels.append(missing)
        elif status == Status.UNKNOWN:
            labels.append(unknown)
    return ", ".join(labels).capitalize()


def _draw_label(
    draw: ImageDraw.ImageDraw, image_size: tuple[int, int], font: ImageFont.FreeTypeFont,
    box: Box, label: str, color: tuple[int, int, int], occupied: list[Box],
) -> None:
    image_width, image_height = image_size
    _, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width = right + 8
    height = bottom - top + 8
    x = max(0, min(round(box.x1), image_width - width))
    start = max(0, min(round(box.y1) - height, image_height - height))
    position = Box(x, start, x + width, start + height)
    for y in range(start, image_height - height + 1, height + 2):
        candidate = Box(x, y, x + width, y + height)
        if all(not candidate.intersection(previous) for previous in occupied):
            position = candidate
            break
    occupied.append(position)
    draw.rectangle((position.x1, position.y1, position.x2, position.y2), fill=color)
    draw.text((x + 4, position.y1 + 4 - top), label, font=font, fill="white")


def _annotate_video(monitor: PPEMonitor, source: Path, output: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not ok:
            raise ValueError(f"Cannot decode video: {source}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot determine video frame rate: {source}")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Cannot open MP4 writer: {output}")
        while ok:
            writer.write(monitor.annotate(frame))
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def process(monitor: PPEMonitor, source: Path, output: Path) -> None:
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Input and output must have different paths")
    suffix, output_suffix = source.suffix.lower(), output.suffix.lower()
    is_image = suffix in IMAGE_SUFFIXES
    if is_image and output_suffix not in IMAGE_SUFFIXES:
        raise ValueError("Image output must use an image extension, for example .jpg")
    if not is_image and (suffix not in VIDEO_SUFFIXES or output_suffix != ".mp4"):
        raise ValueError("Use a supported video input and an .mp4 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        if is_image:
            data = np.fromfile(source, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if frame is None:
                raise ValueError(f"Cannot decode image: {source}")
            ok, encoded = cv2.imencode(output_suffix, monitor.annotate(frame))
            if not ok:
                raise ValueError(f"Cannot encode image: {output}")
            encoded.tofile(temporary)
        else:
            _annotate_video(monitor, source, temporary)
        temporary.replace(output)
