from torch import nn
import torch.nn.functional as F
import torch

from .config import CFG

class XrdEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # XRD feature encoder
        self.densexrd = nn.Linear(5250, 2048)
        self.wtxrd_1 = nn.Linear(2048, 5250)
        self.batch_norm = nn.BatchNorm1d(5250)
        self.wtxrd_2 = nn.Linear(5250, 2048)
        self.wtxrd_3 = nn.Linear(2048, 1024)
        self.lnxrd_1 = nn.LayerNorm(5250)
        self.lnxrd_2 = nn.LayerNorm(1024)

    def forward(self, x):
        projection = x  # residual connection
        x = self.lnxrd_1(x)
        x = F.relu(self.densexrd(x))
        x = F.relu(self.wtxrd_1(x))
        x = self.batch_norm(x)
        x = x + projection  # add residual
        x = F.relu(self.wtxrd_2(x))
        x = F.relu(self.wtxrd_3(x))
        x = self.lnxrd_2(x)
        return x


class latticeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Lattice encoder
        self.densecif_1 = nn.Linear(9, 128)
        self.densecif_2 = nn.Linear(128, 9)
        self.batch_norm = nn.BatchNorm1d(9)
        self.densecif_3 = nn.Linear(9, 256)
        self.wtcif = nn.Linear(256, 64)
        self.lncif_1 = nn.LayerNorm(9)
        self.lncif_2 = nn.LayerNorm(64)

    def forward(self, x):
        projection = x  # residual
        x = self.lncif_1(x)
        x = F.relu(self.densecif_1(x))
        x = F.relu(self.densecif_2(x))
        x = self.batch_norm(x)
        x = x + projection
        x = F.relu(self.densecif_3(x))
        x = self.wtcif(x)
        x = self.lncif_2(x)
        return x


class atomEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Atomic coordinate encoder
        self.densecif_1 = nn.Linear(15, 128)
        self.densecif_2 = nn.Linear(128, 15)
        self.batch_norm = nn.BatchNorm1d(15)
        self.densecif_3 = nn.Linear(15, 256)
        self.wtcif = nn.Linear(256, 64)
        self.lncif_1 = nn.LayerNorm(15)
        self.lncif_2 = nn.LayerNorm(64)

    def forward(self, x):
        projection = x
        x = self.lncif_1(x)
        x = F.relu(self.densecif_1(x))
        x = F.relu(self.densecif_2(x))
        x = self.batch_norm(x)
        x = x + projection
        x = F.relu(self.densecif_3(x))
        x = self.wtcif(x)
        x = self.lncif_2(x)
        return x


class occupyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Occupancy encoder
        self.densecif_1 = nn.Linear(5, 64)
        self.densecif_2 = nn.Linear(64, 5)
        self.batch_norm = nn.BatchNorm1d(5)
        self.densecif_3 = nn.Linear(5, 64)
        self.wtcif = nn.Linear(64, 16)
        self.lncif_1 = nn.LayerNorm(5)
        self.lncif_2 = nn.LayerNorm(16)

    def forward(self, x):
        projection = x
        x = self.lncif_1(x)
        x = F.relu(self.densecif_1(x))
        x = F.relu(self.densecif_2(x))
        x = self.batch_norm(x)
        x = x + projection
        x = F.relu(self.densecif_3(x))
        x = self.wtcif(x)
        x = self.lncif_2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # residual
        x = self.layer_norm(x)
        return x


class CXSPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature, cif_embedding=CFG.cif_embedding, xt_embedding=CFG.xt_embedding):
        super().__init__()
        # Main CXSP model: encoders + projection heads
        self.lattice_encoder = latticeEncoder()
        self.atom_encoder = atomEncoder()
        self.xrd_encoder = XrdEncoder()
        self.occupy_encoder = occupyEncoder()
        self.cif_projection = ProjectionHead(cif_embedding)
        self.xt_projection = ProjectionHead(xt_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Encode CIF components
        lattice_features = self.lattice_encoder(batch[0].float())
        atom_features = self.atom_encoder(batch[1].float())
        occupy_features = self.occupy_encoder(batch[3].float())
        cif_features = torch.cat((lattice_features, atom_features, occupy_features), 1)

        # Encode XRD
        xt_features = self.xrd_encoder(batch[2].float())

        # Project embeddings
        cif_embeddings = self.cif_projection(cif_features)
        xt_embeddings = self.xt_projection(xt_features)

        # Compute similarity logits
        logits = (xt_embeddings @ cif_embeddings.T) / self.temperature
        cifs_similarity = cif_embeddings @ cif_embeddings.T
        xts_similarity = xt_embeddings @ xt_embeddings.T

        # Soft targets for contrastive learning
        targets = F.softmax((cifs_similarity + xts_similarity) / 2 * self.temperature, dim=-1)

        # InfoNCE loss
        xts_loss = F.cross_entropy(logits, targets)
        cifs_loss = F.cross_entropy(logits.T, targets)
        loss = (xts_loss + cifs_loss) / 2.0
        return loss, cifs_loss, xts_loss


def cross_entropy(preds, targets, reduction='none'):
    # Manual cross-entropy for soft labels
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
