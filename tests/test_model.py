"""
Created on 2026-03-07
Copyright (c) 2026 robominds GmbH

Test model helper functions.
"""

import pytest

from ball_detector import model


@pytest.mark.parametrize(
    "architecture",
    [
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fcos_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
        "maskrcnn_resnet50_fpn",
        "maskrcnn_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ],
)
def test_load_from_torch(architecture):
    """Test loading model from torchvision."""
    # Act
    ai_model = model.load_from_torchvision(architecture)

    # Assert
    assert isinstance(ai_model, model.ModelData)
    assert ai_model.name == architecture


@pytest.mark.parametrize(
    ("repo", "model_name"),
    [
        ("ultralytics/yolov5", "yolov5s"),
    ],
)
def test_load_from_torchhub(repo, model_name):
    """Test loading model from torchhub."""
    # Act
    ai_model = model.load_from_torchhub(repo, model_name)

    # Assert
    assert isinstance(ai_model, model.ModelData)
    assert ai_model.name == model_name


@pytest.mark.parametrize(
    "architecture",
    [
        "fasterrcnn_resnet50_fpn",
        "fcos_resnet50_fpn",
        "ssd300_vgg16",
    ],
)
def test_load_from_file(architecture, tmp_path):
    """Load a model from torch, save and try to load from file."""
    # Arrange
    model_data = model.load_from_torchvision(architecture)
    filename = model.filename(tmp_path, architecture)

    # Act
    model.save(filename, model_data.ai_model, model_data)
    model_data = model.load_from_file(filename)

    # Assert
    assert isinstance(model_data, model.ModelData)
    assert model_data.name == architecture
    try:
        model_data.ai_model.eval()
    except AttributeError:
        pytest.fail("Model loading failed!")
