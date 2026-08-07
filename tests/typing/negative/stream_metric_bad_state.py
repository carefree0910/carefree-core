from typing import Dict

from core.learn.schema import IStreamMetric


def merge_bad_state(metric: IStreamMetric[Dict[str, float]]) -> None:
    metric.merge_distributed_states([1])  # expected-mypy: list-item
