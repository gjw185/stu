import torch
import torch.nn as nn
import torch.optim as optim
import os
import pdb
import torchvision
from torch.utils.data import DataLoader, Dataset
from load_data import CIFAR10Dataset, extract_labels_from_filenames
from cnn import SimpleCNN
labels_dict = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}
#train
def train(model, train_loader, test_loader, device):
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次损失
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 每个epoch结束后在测试集上评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        
    torch.save(model.state_dict(), 'cifar10_cnn.pth')

if __name__ == '__main__':
    # 训练集
    train_data_dir = './CIFAR/train'  # 图像存放目录
    train_img = os.listdir(train_data_dir)
    # 构建完整图像路径列表
    train_image_paths = [os.path.join(train_data_dir, fname) for fname in train_img]
    train_label = extract_labels_from_filenames(train_img, labels_dict)
    # 取前1000
    train_image_paths = train_image_paths[:1000]
    train_label = train_label[:1000]

    # 测试集
    test_data_dir = './CIFAR/test'
    test_img = os.listdir(test_data_dir)
    test_image_paths = [os.path.join(test_data_dir, fname) for fname in test_img]
    test_label = extract_labels_from_filenames(test_img, labels_dict)
    test_image_paths = test_image_paths[:1000]
    test_label = test_label[:1000]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()  # 将PIL图像转为Tensor并自动归一化到[0,1]
    ])

    # 创建数据集实例
    train_dataset = CIFAR10Dataset(train_image_paths, train_label, transform=transform)
    test_dataset = CIFAR10Dataset(test_image_paths, test_label, transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #创建模型
    model = SimpleCNN().to(device)

    train(model, train_loader, test_loader, device)