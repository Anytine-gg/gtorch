import torch
from torch import nn
import torchsummary
import torchvision.models
import torchvision.transforms as transforms
import gtorch.utils.misc.plot as gplt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 添加此行


class AlexNet(nn.Module):
    def __init__(self, in_channels, nClasses):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1, stride=1), nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1, stride=1), nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 4096, 6, stride=1), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.layer7 = nn.Sequential(
            nn.Flatten(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )
        self.layer8 = nn.Linear(4096, nClasses)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


def train_a_epoch(idx, model: nn.Module, train_loader, val_loader, loss, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_loss = 0.0
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {idx}")  # 使用tqdm包装train_loader
    for img, label in train_loader_tqdm:
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        l = loss(pred, label)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        epoch_loss += l.item()
        train_loader_tqdm.set_postfix(loss=l.item())  # 在tqdm中显示当前批次的损失值

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f"Validation Epoch {idx}")  # 使用tqdm包装val_loader
        for img, label in val_loader_tqdm:
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            l = loss(pred, label)
            val_loss += l.item()
            _, predicted = torch.max(pred.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            val_loader_tqdm.set_postfix(loss=l.item())  # 在tqdm中显示当前批次的损失值
        accuracy = 100 * correct / total
        print(
            f"Epoch: {idx}, TrainLoss: {epoch_loss/len(train_loader)}, ValLoss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%"
        )

def train_epochs(model, train_dataset, val_dataset, batch_size, nEpochs, device):
    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, False)
    loss = nn.CrossEntropyLoss()
    for idx in range(nEpochs):
        train_a_epoch(idx, model, train_loader, val_loader, loss, device)


if __name__ == "__main__":
    device = 'cudaww'
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST数据集的均值和标准差
        ]
    )
    net = AlexNet(1, 10)
    net.to(device)
    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, transform=transform
    )
    val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform)
    train_epochs(net, train_dataset, val_dataset, 128, 100, device)
    