from torch.utils.data import DataLoader, Dataset
import pdb
import torchvision
from PIL import Image
import os

# 自定义Dataset类，用于加载PNG图像
class CIFAR10Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # 图像文件路径列表
        self.labels = labels  # 对应的标签列表
        self.transform = transform  # 图像变换

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 使用PIL读取图像并确保为RGB格式
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_labels_from_filenames(filenames, labels_dict):
    # 创建一个空字典来存储文件名和对应的标签
    file_labels = []

    # 遍历文件名列表
    for filename in filenames:
        # 分割文件名以提取类别部分
        # 假设文件名格式为 "编号_类别.png"
        parts = filename.split('_')
        if len(parts) > 1:
            category = parts[1].split('.')[0]  # 获取类别名称，去除.png扩展名
            # 获取类别对应的标签
            label = labels_dict.get(category, -1)  # 如果类别不在字典中，返回-1
            file_labels.append(label)

    return file_labels

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

if __name__ == '__main__':

    #训练集
    train_data_dir = './CIFAR/train'  # 图像存放目录
    train_img = os.listdir(train_data_dir)
    # 构建完整图像路径列表
    train_image_paths = [os.path.join(train_data_dir, fname) for fname in train_img]
    train_label = extract_labels_from_filenames(train_img, labels_dict)
    #取前1000
    train_image_paths = train_image_paths[:1000]
    train_label = train_label[:1000]


    #测试集
    test_data_dir = './CIFAR/test'
    test_img = os.listdir(test_data_dir)
    test_image_paths = [os.path.join(test_data_dir, fname) for fname in test_img]
    test_label = extract_labels_from_filenames(test_img, labels_dict)
    test_image_paths = test_image_paths[:1000]
    test_label = test_label[:1000]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()  # 将PIL图像转为Tensor并自动归一化到[0,1]
    ])


    pdb.set_trace()
    # 创建数据集实例
    train_dataset = CIFAR10Dataset(train_image_paths, train_label, transform=transform)
    test_dataset = CIFAR10Dataset(test_image_paths, test_label, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)