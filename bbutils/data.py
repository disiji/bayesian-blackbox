from typing import List, Tuple

import numpy as np


def read_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[int] = []
    scores: List[List[float]] = []
    with open(path, 'r') as f:
        for line in f:
            label_string, *score_strings = line.split()
            label = int(label_string)
            _scores = [float(score) for score in score_strings]
            labels.append(label)
            scores.append(_scores)
    return np.array(labels), np.array(scores)