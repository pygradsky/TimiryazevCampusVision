import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_image_dataloader():
    """Функция для получения DataLoader для изображений."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=r"C:\Users\Professional\PycharmProjects\TimiryazevCampusVision\campuses",
                          transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataset, loader
