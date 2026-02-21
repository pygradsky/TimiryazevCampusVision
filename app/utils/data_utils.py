from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from app.utils.transforms import data_transform
from app.config import CLASSES_DIR, BATCH_SIZE


def get_loaders():
    dataset = datasets.ImageFolder(root=CLASSES_DIR, transform=data_transform)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, dataset.classes
