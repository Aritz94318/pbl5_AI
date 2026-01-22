import pytest
from service_api import aggregate_overall_max, THRESHOLD, LABEL_ID_TO_NAME

def test_max_aggregation():
    probs = [0.1, 0.8, 0.2, 0.3]
    assert aggregate_overall_max(probs) == pytest.approx(0.8)


def test_threshold():
    p = THRESHOLD + 0.01
    pred = 1 if p >= THRESHOLD else 0
    assert LABEL_ID_TO_NAME[pred] == "MALIGNANT"
