import torch
import torch.nn as nn
import torch.nn.functional as F

class X3DFightClassifier(nn.Module):
    """
    X3D-S Video Classifier fine-tuned for fight detection.
    - Architecture: X3D-S (from PyTorchVideo)
    - Inputs: [B, 3, 16, 224, 224] (RGB, normalized 0-1)
    - Output: 2 classes (index 0 = no_fight, index 1 = fight)
    """
    def __init__(self, pretrained=True, num_classes=2):
        super(X3DFightClassifier, self).__init__()
        try:
            self.model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=pretrained)
            in_features = self.model.blocks[-1].proj.in_features
            self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize X3D-S model from PyTorchVideo: {e}")

    def forward(self, x):
        # Input tensor shape: (batch_size, 3, 16, 224, 224)
        return self.model(x)

# Backward-compatibility alias
FightClassifier = X3DFightClassifier

if __name__ == "__main__":
    model = X3DFightClassifier(pretrained=False)
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    output = model(dummy_input)
    print("X3DFightClassifier output shape:", output.shape)
