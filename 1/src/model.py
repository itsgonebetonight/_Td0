from typing import Optional

from PIL import Image
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
except Exception as e:
    torch = None


class EmbeddingExtractor:
    """Extract image embeddings using a pretrained ResNet.

    Produces a 512-d vector (for resnet18). If torch isn't installed, the class
    will raise informative errors when used.
    """

    def __init__(self, device: Optional[str] = None):
        if torch is None:
            raise RuntimeError("PyTorch is required for EmbeddingExtractor. Install torch and torchvision.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        # remove final fc -> get 512-d features
        modules = list(self.model.children())[:-1]
        self.model = torch.nn.Sequential(*modules)
        self.model.eval()
        self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def image_to_embedding(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)
        feat = feat.squeeze().cpu().numpy()
        # Flatten in case shape is (C,1,1)
        return feat.reshape(-1)
