import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataset import LandmarkDataset
from src.models.model import get_model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='train')
    parser.add_argument('--csv_path', type=str, default='data/train.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = LandmarkDataset(
        args.images_dir, args.csv_path, transform=transform)
    num_classes = dataset.data['landmark_id'].nunique()
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'best_model.pth'))
            print("Saved best model.")


if __name__ == '__main__':
    main()
