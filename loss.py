import torch.nn as nn

class Kploss(object):
    def __init__(self):
        self.criterion = nn.MSELoss(reduction="none")

    def __call__(self, logits, targets):
        device = logits.device
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        loss = self.criterion(logits, heatmaps).mean(dim=[2,3])
        loss = torch.sum(loss*kps_weights)/logits.shape[0]
        return loss

