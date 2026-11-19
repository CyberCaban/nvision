"""Match detected PPE to people and assess each item independently."""

from dataclasses import dataclass
from enum import StrEnum
from math import hypot

CLASSES = ("person", "helmet", "vest")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersection(self, other: "Box") -> float:
        return Box(
            max(self.x1, other.x1), max(self.y1, other.y1),
            min(self.x2, other.x2), min(self.y2, other.y2),
        ).area


@dataclass(frozen=True)
class Detection:
    label: str
    box: Box


@dataclass(frozen=True)
class Pose:
    box: Box
    keypoints: tuple[tuple[float, float, float], ...]


class Status(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Assessment:
    person: Box
    helmet: Status
    vest: Status


def _match_poses(people: list[Box], poses: list[Pose]) -> dict[int, Pose]:
    candidates = []
    for i, person in enumerate(people):
        for j, pose in enumerate(poses):
            overlap = person.intersection(pose.box)
            union = person.area + pose.box.area - overlap
            if union and overlap / union >= 0.5:
                candidates.append((overlap / union, i, j))
    matches = {}
    used = set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matches and j not in used:
            matches[i] = poses[j]
            used.add(j)
    return matches


def _visibility(pose: Pose | None, kind: str, width: int, height: int) -> bool:
    if pose is None:
        return False
    # COCO pose indices: face 0–4, shoulders 5–6, hips 11–12.
    visible = {
        i for i, (x, y, confidence) in enumerate(pose.keypoints)
        if confidence >= 0.6 and 2 <= x < width - 2 and 2 <= y < height - 2
    }
    if kind == "helmet":
        return pose.box.y1 > 2 and len(visible.intersection(range(5))) >= 2
    return {5, 6, 11, 12} <= visible


def _zone(person: Box, kind: str, pose: Pose | None) -> Box:
    width, height = person.x2 - person.x1, person.y2 - person.y1
    if kind == "helmet":
        if pose:
            face = [(x, y) for x, y, confidence in pose.keypoints[:5] if confidence >= 0.6]
            if len(face) >= 2:
                x = sum(point[0] for point in face) / len(face)
                y = sum(point[1] for point in face) / len(face)
                return Box(x - width * 0.35, y - height * 0.24,
                           x + width * 0.35, y + height * 0.06)
        return Box(person.x1, person.y1 - height * 0.08,
                   person.x2, person.y1 + height * 0.25)
    if pose and len(pose.keypoints) >= 13:
        torso = [pose.keypoints[index] for index in (5, 6, 11, 12)]
        if all(confidence >= 0.6 for _, _, confidence in torso):
            return Box(min(x for x, _, _ in torso) - width * 0.12,
                       min(y for _, y, _ in torso) - height * 0.05,
                       max(x for x, _, _ in torso) + width * 0.12,
                       max(y for _, y, _ in torso) + height * 0.05)
    return Box(person.x1, person.y1 + height * 0.16,
               person.x2, person.y1 + height * 0.62)


def assess(
    detections: list[Detection], poses: list[Pose], width: int, height: int,
) -> list[Assessment]:
    people = [item.box for item in detections if item.label == "person" and item.box.area]
    pose_by_person = _match_poses(people, poses)
    statuses = []
    for kind in CLASSES[1:]:
        zones = [_zone(person, kind, pose_by_person.get(i)) for i, person in enumerate(people)]
        present, ambiguous = set(), set()
        for item in detections:
            if item.label != kind or not item.box.area:
                continue
            x, y = item.box.center
            scores = []
            for i, zone in enumerate(zones):
                coverage = zone.intersection(item.box) / item.box.area
                if coverage >= 0.5 and zone.contains(x, y):
                    zx, zy = zone.center
                    distance = hypot((x - zx) / (zone.x2 - zone.x1),
                                     (y - zy) / (zone.y2 - zone.y1))
                    scores.append((coverage - 0.5 * distance, i))
            if scores:
                scores.sort(reverse=True)
                best, owner = scores[0]
                contenders = [i for score, i in scores if best - score < 0.10]
                if len(contenders) == 1:
                    present.add(owner)
                else:
                    ambiguous.update(contenders)
        item_statuses = []
        for i, zone in enumerate(zones):
            if i in present:
                status = Status.PRESENT
            else:
                status = Status.UNKNOWN
                if i not in ambiguous and _visibility(pose_by_person.get(i), kind, width, height):
                    overlaps = any(
                        i != j and zone.intersection(other) >= 0.45 * zone.area
                        for j, other in enumerate(people)
                    )
                    if not overlaps:
                        status = Status.MISSING
            item_statuses.append(status)
        statuses.append(item_statuses)
    return [Assessment(person, helmet, vest) for person, helmet, vest in zip(people, *statuses)]
