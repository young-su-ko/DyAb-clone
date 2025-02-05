import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import Resize


class DyAb(nn.Module):
    def __init__(self, img_dim, device):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, 1)
        self.img_dim = int(img_dim)
        self.device = device
        self._resize = Resize((self.img_dim, self.img_dim))

    def forward(self, ab_1, ab_2):
        # ab_1 [batch, len, hidden_dim]
        # ab_2 [batch, len, hidden_dim]

        B = ab_1.shape[0]
        input_image = torch.zeros((B, 3, self.img_dim, self.img_dim), device=self.device)

        embedding_diffs = ab_1 - ab_2

        resized_diffs = self._resize(embedding_diffs)
        resized_diffs -= torch.amin(resized_diffs)

        # check that embeddings are not all zeros
        if torch.amax(resized_diffs) > 0:
            resized_diffs /= torch.amax(resized_diffs)
        resized_diffs = self._resize(resized_diffs)
        
        # only first RGB channel has values, rest are 0s.
        input_image[:, 0, :, :] += resized_diffs

        predicted_delta = self.resnet(input_image).squeeze().float()

        return predicted_delta



        
