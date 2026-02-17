import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

BASE_PATH = r"C:\Users\Professional\PycharmProjects\TimiryazevCampusVision\campuses"


def get_image_dataloader():
    """Функция для получения DataLoader для изображений."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    try:
        dataset = ImageFolder(root=BASE_PATH, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    except FileNotFoundError:
        print(f"Одна из папок пустая или не содержит в себе допустимые форматы изображений (.jpg, .jpeg, .png, ...)!\n"
              f"Проверьте структуру папок и наличие изображений в подпапках '{BASE_PATH}'.")
        return None, None

    return dataset, loader
