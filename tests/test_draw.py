"""
Created on 2026-03-10
Copyright (c) 2026 Munich University of Applied Sciences
"""

import pandas as pd
import pytest

from ball_detector import draw


@pytest.mark.parametrize(
    "n",
    [
        (2),
        (3),
    ],
)
def test_colors(n):
    """Test if colors creates a dict of tuples."""
    # Arrange
    df = pd.DataFrame([{"category_id": 1}, {"category_id": 2}, {"category_id": 3}])

    # Act
    colors = draw.colors(n)
    df["colors"] = df["category_id"].map(colors)

    # Assert
    assert isinstance(colors, dict)
    assert all(isinstance(color, tuple) for color in colors.values())
    assert df[df["category_id"] == 1]["colors"].iloc[0] == colors[1]
