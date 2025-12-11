import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import itertools
from tqdm.autonotebook import tqdm
import torch
import multiprocessing
import time
from model.cxsp_utils import AvgMeter, get_lr
from model.config import CFG
from model.dataset import CXSPDataset
from model.cxsp_model import CXSPModel


def make_train_valid_dfs():
    # Load all file names from captions folder
    dataframe = os.listdir(CFG.captions_path)
    max_id = len(dataframe) if not CFG.debug else 100
    np.random.seed(512)

    image_ids = np.arange(0, max_id)  # sequential IDs

    # Random 20% for validation split
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )

    # 80% train, 20% valid
    train_dataframe = [dataframe[id_] for id_ in image_ids if id_ not in valid_ids]
    valid_dataframe = [dataframe[id_] for id_ in image_ids if id_ in valid_ids]
    return train_dataframe, valid_dataframe


def build_loaders(mode):
    # Create dataset and dataloader
    dataset = CXSPDataset(
        mode,
        transforms=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in tqdm_object:
        # Forward pass
        lossall = model(batch)
        loss = lossall[0]

        optimizer.zero_grad()      # reset gradients
        loss.backward()            # backward pass
        optimizer.step()           # update weights

        if step == "batch":
            lr_scheduler.step()    # batch-level LR scheduling

        count = batch[0].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter, lossall[1].item(), lossall[2].item()


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    for batch in tqdm_object:
        # Validation forward pass (no gradients)
        lossall = model(batch)
        loss = lossall[0]

        count = batch[0].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter, lossall[1].item(), lossall[2].item()


def main():
    multiprocessing.set_start_method('spawn', force=True)

    # Set random seed for reproducibility
    seed = 512
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build dataloaders
    train_loader = build_loaders(mode="train")
    valid_loader = build_loaders(mode="valid")

    model = CXSPModel().to(CFG.device)

    # Separate LR for different model parts
    params = [
        {"params": model.lattice_encoder.parameters(), "lr": CFG.cif_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": model.atom_encoder.parameters(), "lr": CFG.cif_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": model.xrd_encoder.parameters(), "lr": CFG.xrd_encoder_lr, "weight_decay": CFG.weight_decay},
        {"params": itertools.chain(
            model.cif_projection.parameters(), model.xt_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)

    # LR scheduler: reduce LR if validation loss stops improving
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    early_stop_patience = 50       # early stopping threshold
    num_epochs_no_improve = 0
    step = "epoch"
    best_loss = float('inf')
    start2 = time.time()

    for epoch in range(CFG.epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        # Save best model based on validation loss
        if valid_loss[0].avg < best_loss:
            num_epochs_no_improve = 0
            best_loss = valid_loss[0].avg
            torch.save(model.state_dict(), "best.pt")
            print(str(epoch))
            print(f"time: {(time.time()-start2)/3600} hours, Saved Best Model!")
        else:
            num_epochs_no_improve += 1
            print(f"{epoch}_epochs_time: {(time.time()-start2)/3600} hours,")

        # Early stopping check
        if num_epochs_no_improve == early_stop_patience:
            break


if __name__ == '__main__':
    start = time.time()
    main()
