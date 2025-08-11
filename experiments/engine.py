from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import dice_coefficient


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    bce_weight: float = 0.5) -> Tuple[float, float]:
    model.train()
    ce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    steps = 0

    for batch in dataloader:
        if batch is None:
            continue
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss_bce = ce(logits, masks)
        probs = torch.sigmoid(logits)
        loss_dice = 1.0 - dice_coefficient(probs, masks)
        loss = bce_weight * loss_bce + (1.0 - bce_weight) * loss_dice
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dice = dice_coefficient(probs, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        steps += 1

    avg_loss = total_loss / steps if steps else 0.0
    avg_dice = total_dice / steps if steps else 0.0
    return avg_loss, avg_dice


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss_bce = ce(logits, masks)
            probs = torch.sigmoid(logits)
            loss_dice = 1.0 - dice_coefficient(probs, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            dice = dice_coefficient(probs, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            steps += 1

    avg_loss = total_loss / steps if steps else 0.0
    avg_dice = total_dice / steps if steps else 0.0
    return avg_loss, avg_dice
