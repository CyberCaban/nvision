"""Command line entry point for dataset preparation, training and inference."""

import argparse
import json
import os
from pathlib import Path

from ppe.dataset import prepare_dataset


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _probability(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construction PPE detection with YOLOv8")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert an extracted YOLO dataset to three classes")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/ppe"))

    train = commands.add_parser("train", help="Fine-tune the YOLOv8n detector")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--epochs", type=_positive, default=50)
    train.add_argument("--batch", type=_positive, default=8)
    train.add_argument("--project", type=Path, default=Path("runs/ppe"))
    train.add_argument("--name", default="train")

    evaluate = commands.add_parser("evaluate", help="Evaluate a detector on the held-out test set")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--batch", type=_positive, default=8)

    predict = commands.add_parser("predict", help="Annotate a photo or video file")
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument("--source", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--confidence", type=_probability, default=0.35)
    for command in (train, evaluate, predict):
        command.add_argument("--device", help="Pass cpu or a CUDA device index, e.g. 0")
        command.add_argument("--imgsz", type=_positive, default=640)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare_dataset(args.config, args.output))
            return
        for field in ("data", "weights"):
            path = getattr(args, field, None)
            if path is not None and not path.is_file():
                raise ValueError(f"Missing {field} file: {path}")

        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("runs/.ultralytics").resolve()))
        if args.command == "predict":
            from ppe.inference import PPEMonitor, process

            process(PPEMonitor(args.weights, args.device, args.confidence, args.imgsz),
                    args.source, args.output)
            print(args.output.resolve())
            return

        from ultralytics import YOLO

        options = dict(data=str(args.data.resolve()), device=args.device, imgsz=args.imgsz,
                       batch=args.batch, workers=0)
        if args.command == "train":
            model = YOLO("yolov8n.pt")
            model.train(**options, project=str(args.project.resolve()), name=args.name,
                        epochs=args.epochs, seed=42)
            print(Path(model.trainer.best).resolve())
        else:
            metrics = YOLO(str(args.weights)).val(
                **options, project=str(Path("runs/ppe").resolve()), split=args.split, name="evaluation",
            )
            report = {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "classes": {
                    metrics.names[index]: dict(zip(("precision", "recall", "AP50", "AP50-95"),
                                                   map(float, metrics.box.class_result(i))))
                    for i, index in enumerate(metrics.box.ap_class_index)
                },
            }
            report_path = Path(metrics.save_dir) / "metrics.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(report_path.resolve())
    except ModuleNotFoundError as error:
        if error.name in {"ultralytics", "cv2", "numpy"}:
            parser.error("Install vision dependencies with: uv sync --extra vision")
        raise
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
