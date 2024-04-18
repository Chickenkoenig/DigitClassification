import torch
from model import ImageClassifier
from train import train, validate
from data_loader import get_data_loaders
import argparse
import wandb
from dataclasses import dataclass
import hydra
from omegaconf import OmegaConf
@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float


@hydra.main(config_path="../config", config_name="train", version_base="1.1")
def main(cfg):
    config = OmegaConf.to_object(cfg)
    train_config = TrainConfig(**config)

    wandb.init(project='mnist_classification',config = {
        "learning_rate": train_config.learning_rate,
        "epochs": train_config.epochs,
        "batch_size": train_config.batch_size
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(batch_size=train_config.batch_size)
    model = ImageClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    wandb.watch(model, log='all')
    for epoch in range(train_config.epochs):
        train_loss = train(model, device,train_loader, optimizer, criterion)
        val_loss = validate(model, device, val_loader, criterion)
        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}')

    wandb.finish()


if __name__ == "__main__":
    main()