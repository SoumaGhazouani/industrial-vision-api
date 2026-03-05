from torch.utils.data import DataLoader
from torchvision import transforms
from src.vision.dataset import MVTecBinaryDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = MVTecBinaryDataset(
    root_dir="data/mvtec/bottle",
    split="train",
    transform=transform
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

for images, labels in loader:
    print(images.shape) #[batch_size, channels, height, width]
    print(labels.shape)
    break