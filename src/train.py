import torch
import torch.nn as nn
from torch.optim import Adam
from model import ImageClassifier
from data_loader import get_data_loaders
import wandb

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({'train_loss': loss.item()})
    return total_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            wandb.log({'val_loss': loss.item()})
    return total_loss / len(val_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders()
    model = ImageClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss = validate(model, device, val_loader, criterion)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == "__main__":
    main()