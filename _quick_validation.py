import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
import random

random.seed(0)
torch.manual_seed(0)

root_dir = 'data/UTKface'
all_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

targets = [
    '9_1_0_20170109204626343.jpg.chip.jpg',
    '36_1_1_20170109141849605.jpg.chip.jpg',
    '80_1_0_20170110140603775.jpg.chip.jpg',
    '54_0_0_20170111210520573.jpg.chip.jpg',
    '42_0_2_20170104184350086.jpg.chip.jpg',
]

# 속도를 위해 서브셋만 사용 (train/val 소속은 실제 노트북과 동일하게 유지)
train_sub = [f for f in train_files if f not in targets]
val_sub = [f for f in val_files if f not in targets]
random.shuffle(train_sub)
random.shuffle(val_sub)
train_sub = train_sub[:3000] + [f for f in targets if f in train_files]
val_sub = val_sub[:600] + [f for f in targets if f in val_files]

print(f'train subset: {len(train_sub)}, val subset: {len(val_sub)}')
for t in targets:
    where = 'train' if t in train_files else 'val'
    print(f'  {t} -> {where}')

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = file_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        age = int(self.image_files[idx].split('_')[0])
        if self.transform:
            image = self.transform(image)
        return image, age

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = UTKFaceDataset(root_dir, train_sub, transform=train_transform)
val_dataset = UTKFaceDataset(root_dir, val_sub, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class AgeEstimationModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if not (name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc')):
                    param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)

device = torch.device('cpu')
model = AgeEstimationModel(freeze_backbone=True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

num_epochs = 3
best_val_loss = float('inf')
save_path = '_quick_validation_best.pth'

for epoch in range(num_epochs):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  time={time.time()-t0:.1f}s', flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f'  -> best model saved (val_loss={val_loss:.4f})', flush=True)

    scheduler.step(val_loss)

# 베스트 체크포인트로 타깃 이미지들 예측
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

print('\n=== 타깃 이미지 예측 결과 (베스트 체크포인트) ===')
for t in targets:
    actual = int(t.split('_')[0])
    img = Image.open(os.path.join(root_dir, t)).convert('RGB')
    x = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = round(model(x).item())
    where = 'train' if t in train_files else 'val'
    print(f'{t}  [{where}]  실제={actual}  예측={pred}  오차={abs(actual-pred)}')
