import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import itertools
from tqdm.autonotebook import tqdm
import torch
import multiprocessing
import time
import torchvision.transforms as transforms
from model.xtsp_utils import AvgMeter, get_lr
from model.config import CFG
from model.dataset import XTSPDataset
from model.xtsp_model import XTSPModel


def make_train_valid_dfs():
    """Split dataset into train and validation lists."""
    files = os.listdir(CFG.captions_path)
    max_id = len(files) if not CFG.debug else 100
    np.random.seed(512)
    ids = np.arange(0, max_id)
    valid_ids = np.random.choice(ids, size=int(0.2 * len(ids)), replace=False)
    train_files = [files[i] for i in ids if i not in valid_ids]
    valid_files = [files[i] for i in ids if i in valid_ids]
    return train_files, valid_files


def build_loaders(mode):
    """Build DataLoader for a given mode (train/valid/test)."""
    dataset = XTSPDataset(mode, transforms=transforms)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=(mode == "train"),
    )
    return loader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """Run one training epoch."""
    loss_meter = AvgMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in tqdm(train_loader, total=len(train_loader)):
        lossall = model(batch)
        loss = lossall[0]

        optimizer.zero_grad()       # Clear gradients
        loss.backward()             # Backprop
        optimizer.step()            # Update parameters
        if step == "batch":
            lr_scheduler.step()     # Update LR per batch

        loss_meter.update(loss.item(), batch[0].size(0))
        tqdm.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter, lossall[1].item(), lossall[2].item()


def valid_epoch(model, valid_loader):
    """Run one validation epoch."""
    loss_meter = AvgMeter()
    for batch in tqdm(valid_loader, total=len(valid_loader)):
        lossall = model(batch)
        loss = lossall[0]
        loss_meter.update(loss.item(), batch[0].size(0))
        tqdm.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter, lossall[1].item(), lossall[2].item()


def main():
    """Main training loop with early stopping."""
    multiprocessing.set_start_method('spawn', force=True)
    seed = 512
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = build_loaders("train")
    valid_loader = build_loaders("valid")

    model = XTSPModel().to(CFG.device)
    
    # Set different LR for different modules
    params = [
        {"params": model.lattice_encoder.parameters(), "lr": CFG.cif_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": model.atom_encoder.parameters(), "lr": CFG.cif_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": model.xrd_encoder.parameters(), "lr": CFG.xrd_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": itertools.chain(model.cif_projection.parameters(), model.xt_projection.parameters()),
         "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    early_stop_patience = 50
    num_epochs_no_improve = 0
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(CFG.epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step="epoch")
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        # Save best model
        if valid_loss[0].avg < best_loss:
            best_loss = valid_loss[0].avg
            num_epochs_no_improve = 0
            torch.save(model.state_dict(), "best.pt")
            print(f"Epoch {epoch}: Best model saved. Time: {(time.time()-start_time)/3600:.2f}h")
        else:
            num_epochs_no_improve += 1
            print(f"Epoch {epoch}: No improvement. Time: {(time.time()-start_time)/3600:.2f}h")

        # Early stopping
        if num_epochs_no_improve >= early_stop_patience:
            break


if __name__ == '__main__':
    start = time.time()
    main()
