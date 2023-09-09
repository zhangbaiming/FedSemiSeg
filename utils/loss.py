import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, _, _ = target.size(0), target.size(1), target.size(2)
        smooth = 1e-20  # 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1).float()

        intersection = input_flat * target_flat

        # loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i, :, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.filters = [2048, 1024, 512, 256]
        self.projectors = [self.Projector(dim) for dim in self.filters]

    class Projector(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            mid_dim = in_dim // 8
            self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
            self.norm1 = nn.BatchNorm2d(mid_dim)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(mid_dim, in_dim, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            return x

    def contrastive_loss(self, out, pseudo_label, image_prototype, device, tem=0.1):
        _, c, w, h = out.shape

        # target = pseudo_label[:, None, :, :]
        # target = F.interpolate(pseudo_label.float(), (w, h)) > 0.5
        target = F.adaptive_avg_pool2d(pseudo_label.unsqueeze(1).float(), (w, h)) > 0.5

        target = target.repeat(1, c, 1, 1).permute(0, 2, 3, 1)
        fg_proto = torch.masked_select(out.permute(0, 2, 3, 1), target).view(-1, c).mean(dim=0, keepdim=True)
        if torch.isnan(fg_proto[0][0]):
            return fg_proto[0][0]

        proto = F.normalize(fg_proto, dim=1)

        # pos = torch.Tensor(image_prototype[0]).cuda()
        # neg = torch.Tensor(image_prototype[1]).cuda()
        pos = image_prototype[0]
        neg = image_prototype[1]
        pos = F.normalize(pos, dim=1)
        neg = F.normalize(neg, dim=1)

        pos_sim = torch.matmul(proto, pos.t())
        neg_sim = torch.matmul(proto, neg.t())

        pos_sim = pos_sim.mean(dim=1, keepdim=True)
        sim = torch.cat([pos_sim, neg_sim], dim=1)
        sim /= tem
        labels = torch.zeros(sim.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(sim, labels, reduction='sum')
        loss /= sim.shape[0]

        return loss

    def forward(self, out, pseudo_label, image_prototype, device):
        _, c, w, h = out.shape

        index = self.filters.index(c)
        projector = self.projectors[index].to(device)

        out = projector(out)
        return self.contrastive_loss(out, pseudo_label, image_prototype, device)
