import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

class SLIP(nn.Module):
    def __init__(self, tokenizer, img_encoder, text_encoder) -> None:
        super().__init__()
        self.global_crop = None
        self.aug = None
        self.tokenizer = tokenizer

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder

        self.simclr_projector = nn.Identity()
        self.img_projector = nn.Identity()
        self.text_projector = nn.Identity()

    def forward(self, img, text):
        xi, x1, x2 = self.crop(img), self.aug(img), self.aug(img)
        yt = self.tokenizer(text)

        wi, w1, w2 = self.img_encoder(xi, x1, x2)
        wt = self.text_encoder(yt)

        z1, z2 = self.simclr_projector(w1), self.simclr_projector(w2)
        zi, zt = self.img_projector(wi), self.text_projector(wt)

        return z1, z2, zi, zt

class SLIPLoss(nn.Module):
    def __init__(self, c, tau, s) -> None:
        super().__init__()
        self.tau = tau
        self.s = s
        self.c = c # simCLR loss scale

        self.cross_entropy = nn.CrossEntropyLoss()

    def simclr_loss(self, z1, z2):
        N, _ = z1.shape
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        label = range(N)
        mask = torch.eye(N) * 1e9

        logit = z1 @ z2.T
        logit1 = z1 @ z1.T - mask
        logit2 = z2 @ z2.T - mask
        
        logit1 = torch.cat(logit, logit1)
        logit2 = torch.cat(logit.T, logit2)
        
        l1 = self.cross_entropy(logit1 / self.tau, label)
        l2 = self.cross_entropy(logit2 / self.tau, label)
        
        loss = (l1 + l2) / 2
        
        return loss        

    def clip_loss(self, zi, zt):
        N, _ = zi.shape
        zi = F.normalize(zi)
        zt = F.normalize(zt)
        
        label = range(N)
        logit = torch.exp(self.s) * zi @ zt.T
        li = self.cross_entropy(logit, label)
        lt = self.cross_entropy(logit.T, label)
        
        loss = (li + lt) / 2

        return loss

    def forward(self, z1, z2, zi, zt):
        return self.c * self.simclr_loss(z1, z2) * self.clip_loss(zi, zt)