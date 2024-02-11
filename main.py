import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from my_cnn import MyCNN

data_transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.Resize((20, 20)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root="data/train", transform=data_transform)

dataloader_train = DataLoader(dataset_train, batch_size=24, shuffle=True)

model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(40):
    for batch_idx, (data, target) in enumerate(dataloader_train):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}/{10}, Batch {batch_idx}/{len(dataloader_train)}, Loss: {loss.item()}")

# Save the trained model if needed
torch.save(model.state_dict(), 'model.pth')



