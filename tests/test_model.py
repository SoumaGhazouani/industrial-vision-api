import torch
from src.vision.model import VisionModel


def test_model():

    model = VisionModel()

    dummy = torch.randn(1, 3, 224, 224)

    out = model(dummy)

    print(out.shape)


if __name__ == "__main__":
    test_model()