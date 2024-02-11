import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from my_cnn import MyCNN


data_transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.Resize((20, 20)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_test = datasets.ImageFolder(root="data/test", transform=data_transform)

dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

model = MyCNN()
model.load_state_dict(torch.load("model.pth"))

model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in dataloader_test:
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Accuracy on the test dataset: {accuracy * 100:.2f}%")