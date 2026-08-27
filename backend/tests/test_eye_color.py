"""Tests for the eye-color heuristic.

The HSV classifier is tested directly on synthetic values (pure unit test);
the landmark path is exercised in the real-data sanity script, not here, to
keep the suite offline-friendly.
"""

from app.models.eye_color_heuristic import EYE_COLORS_FR, classify_iris_hsv


def test_classify_desaturated_bright_is_gray():
    assert classify_iris_hsv(h=100, s=20, v=150) == "gray"


def test_classify_desaturated_dark_is_brown():
    assert classify_iris_hsv(h=100, s=20, v=60) == "brown"


def test_classify_blue():
    assert classify_iris_hsv(h=110, s=120, v=140) == "blue"


def test_classify_green():
    assert classify_iris_hsv(h=60, s=100, v=120) == "green"


def test_classify_hazel():
    assert classify_iris_hsv(h=30, s=90, v=110) == "hazel"


def test_classify_brown_low_hue_high_saturation():
    assert classify_iris_hsv(h=12, s=160, v=90) == "brown"


def test_all_labels_have_french_display():
    for h in range(0, 180, 5):
        for s in (10, 80, 200):
            for v in (50, 120, 220):
                assert classify_iris_hsv(h, s, v) in EYE_COLORS_FR
