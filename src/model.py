import torch


class SNN(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def similarity_score(imgs1, imgs2):
        return (torch.nn.CosineSimilarity(dim=1)(imgs1, imgs2) + 1) / 2

    def forward(self, imgs1, imgs2):
        imgs1 = self.backbone(imgs1)
        imgs2 = self.backbone(imgs2)
        output = self.similarity_score(imgs1, imgs2)

        return output
