import torch
import torch.nn as nn
import numpy as np

class LightweightUNet(nn.Module):
    """
    Lightweight U-Net Convolutional Neural Network (CNN) architecture
    for pixel-wise semantic segmentation of satellite canopy imagery.
    Input Channels: 5 (B2-Blue, B3-Green, B4-Red, B8-NIR, NDVI)
    Output: Single probability map (0.0 to 1.0) per pixel.
    """
    def __init__(self, in_channels=5, out_channels=1):
        super(LightweightUNet, self).__init__()
        
        # Encoder (Feature Extraction)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Segmentation Map Generation)
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through CNN
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        b = self.bottleneck(p1)
        up = self.upconv(b)
        # Skip connection concat
        concat = torch.cat([up, x1], dim=1)
        out = self.dec1(concat)
        return self.sigmoid(out)


class CanopyClassifier:
    """
    Inference Engine class to evaluate satellite tensor patches.
    """
    def __init__(self):
        # Automatically use GPU if available, otherwise fallback to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LightweightUNet(in_channels=5, out_channels=1).to(self.device)
        self.model.eval() # Set model to evaluation mode

    def predict_canopy(self, tensor_patch: np.ndarray, ndvi_threshold: float = 0.35) -> np.ndarray:
        """
        Receives a numpy patch array of shape (5, H, W)
        Returns a binary array (H, W) where 1 = Canopy, 0 = Non-Canopy
        """
        with torch.no_grad():
            # Add batch dimension: (5, H, W) -> (1, 5, H, W)
            input_tensor = torch.from_numpy(tensor_patch).unsqueeze(0).float().to(self.device)
            prob_map = self.model(input_tensor).squeeze().cpu().numpy()
            
        # Extract NDVI channel for rule verification
        ndvi_layer = tensor_patch[4]
        
        # Binary Canopy Mask combining CNN output (> 0.5 probability) & NDVI threshold verification
        binary_mask = (prob_map >= 0.5) & (ndvi_layer >= ndvi_threshold)
        return binary_mask.astype(np.uint8)


# Self-test block to verify model initialization and forward pass
if __name__ == "__main__":
    print("Testing CNN Model Pipeline...")
    classifier = CanopyClassifier()
    print(f"✅ CNN PyTorch Model initialized on device: {classifier.device}")

    # Create dummy satellite patch [5 bands: B2, B3, B4, B8, NDVI] of shape 256x256
    dummy_patch = np.random.rand(5, 256, 256).astype(np.float32)
    # Set high dummy NDVI in top-left corner
    dummy_patch[4, 0:100, 0:100] = 0.60 

    predicted_mask = classifier.predict_canopy(dummy_patch, ndvi_threshold=0.35)
    print(f"✅ Predict call success! Output mask shape: {predicted_mask.shape}")
    print(f"✅ Detected Canopy Pixels in dummy test: {np.sum(predicted_mask == 1)}")