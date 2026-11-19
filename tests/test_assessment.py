import pytest

from ppe.assessment import Assessment, Box, Detection, Pose, Status, assess

PERSON = Box(10, 10, 110, 210)
HELMET = Box(43, 10, 77, 27)
VEST = Box(36, 58, 84, 110)


def pose(box=PERSON, *, face=True, torso=True):
    offset = box.x1 - PERSON.x1
    points = [(0.0, 0.0, 0.0)] * 17
    for index, (x, y) in {0: (60, 29), 1: (56, 26), 2: (64, 26)}.items():
        points[index] = (x + offset, y, 0.9 if face else 0.1)
    for index, (x, y) in {5: (37, 55), 6: (83, 55), 11: (41, 116), 12: (79, 116)}.items():
        points[index] = (x + offset, y, 0.9 if torso else 0.1)
    return Pose(box, tuple(points))


@pytest.mark.parametrize(("equipment", "helmet", "vest"), [
    ([], Status.MISSING, Status.MISSING),
    ([Detection("helmet", HELMET)], Status.PRESENT, Status.MISSING),
    ([Detection("vest", VEST)], Status.MISSING, Status.PRESENT),
    ([Detection("helmet", HELMET), Detection("vest", VEST)], Status.PRESENT, Status.PRESENT),
])
def test_assesses_each_item(equipment, helmet, vest):
    detections = [Detection("person", PERSON), *equipment]
    assert assess(detections, [pose()], 300, 250) == [Assessment(PERSON, helmet, vest)]


def test_helmet_held_at_torso_does_not_count():
    detections = [Detection("person", PERSON), Detection("helmet", Box(45, 70, 75, 95))]
    assert assess(detections, [pose()], 300, 250)[0].helmet == Status.MISSING


def test_does_not_share_helmet_between_workers():
    other = Box(55, 10, 155, 210)
    detections = [Detection("person", PERSON), Detection("person", other), Detection("helmet", HELMET)]
    results = assess(detections, [pose(other), pose()], 300, 250)
    assert [person.helmet for person in results] == [Status.PRESENT, Status.MISSING]


def test_ambiguous_equipment_marks_both_workers_unknown():
    other = Box(30, 10, 130, 210)
    between = Box(53, 10, 87, 27)
    detections = [Detection("person", PERSON), Detection("person", other), Detection("helmet", between)]
    results = assess(detections, [pose(other), pose()], 300, 250)
    assert [person.helmet for person in results] == [Status.UNKNOWN, Status.UNKNOWN]


def test_overlapping_people_make_an_absent_detection_inconclusive():
    other = Box(24, 10, 124, 210)
    detections = [Detection("person", PERSON), Detection("person", other)]
    assert assess(detections, [pose(other), pose()], 300, 250) == [
        Assessment(PERSON, Status.UNKNOWN, Status.UNKNOWN),
        Assessment(other, Status.UNKNOWN, Status.UNKNOWN),
    ]


@pytest.mark.parametrize(("keypoints", "helmet", "vest"), [
    ([], Status.UNKNOWN, Status.UNKNOWN),
    ([pose(face=False)], Status.UNKNOWN, Status.MISSING),
    ([pose(torso=False)], Status.MISSING, Status.UNKNOWN),
    ([pose(Box(10, 0, 110, 210))], Status.UNKNOWN, Status.MISSING),
])
def test_visibility_limits_missing_status(keypoints, helmet, vest):
    assert assess([Detection("person", PERSON)], keypoints, 300, 250) == [
        Assessment(PERSON, helmet, vest),
    ]


def test_frame_without_people():
    assert assess([Detection("helmet", HELMET)], [], 300, 250) == []
