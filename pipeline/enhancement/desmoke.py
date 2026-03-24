"""
CycleGAN-based smoke removal for laparoscopic frames.
Trained on unpaired clear/hazy laparoscopic surgery images.

At inference, only G_hazy2clear.pth is needed.
Training requires all four weight files.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Building blocks

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection


class Generator(nn.Module):
    """
    ResNet-4 generator.
    Encoder → 4 residual blocks → Decoder.
    """
    def __init__(self, in_channels=3, features=64, n_residuals=4):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(features * 2, features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(n_residuals)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features * 2, features, 3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(features, in_channels, 7),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        return self.decoder(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    Classifies overlapping image patches as real or fake.
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, features,      normalize=False),
            *block(features,    features * 2),
            *block(features * 2, features * 4),
            *block(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, 4, padding=1),  # patch output
        )

    def forward(self, x):
        return self.model(x)


# Transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert generator output tensor back to BGR uint8 for OpenCV."""
    img = tensor.squeeze(0).cpu().detach()
    img = (img * 0.5 + 0.5).clamp(0, 1)          # [-1,1] → [0,1]
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



# Inference


def load_generator(weights_path: str = "weights/G_hazy2clear_lite.pth", device: str = None) -> Generator:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(device)
    G.load_state_dict(torch.load(weights_path, map_location=device))
    G.eval()
    return G


def desmoke_frame(G: Generator, frame: np.ndarray,
                  device: str = None) -> np.ndarray:
    """
    Remove smoke from a single BGR frame using the trained generator.
    Returns BGR uint8 frame at original resolution.

    frame: BGR uint8 numpy array (OpenCV format)
    """
    if device is None:
        device = next(G.parameters()).device

    original_h, original_w = frame.shape[:2]

    # Preprocess
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        output = G(tensor)

    # Postprocess — resize back to original resolution
    result = tensor_to_frame(output)
    result = cv2.resize(result, (original_w, original_h),
                        interpolation=cv2.INTER_LINEAR)
    return result