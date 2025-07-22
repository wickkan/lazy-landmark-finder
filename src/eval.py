import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataset import LandmarkDataset
from src.models.model import get_model
import numpy as np
from tqdm import tqdm


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='train')
    parser.add_argument('--csv_path', type=str,
                        default='data/train_000_remap.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model_path', type=str,
                        default='models/best_model.pth')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = LandmarkDataset(
        args.images_dir, args.csv_path, transform=transform)
    num_classes = dataset.data['landmark_id'].nunique()
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    top1s, top5s = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1s.append(top1)
            top5s.append(top5)
    print(f"Top-1 Accuracy: {np.mean(top1s):.2f}%")
    print(f"Top-5 Accuracy: {np.mean(top5s):.2f}%")


if __name__ == '__main__':
    main()
