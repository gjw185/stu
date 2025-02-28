import torch
from cnn import SimpleCNN
from PIL import Image
import torchvision

if __name__ == '__main__':
    model = SimpleCNN()
    model.load_state_dict(torch.load('./cifar10_cnn.pth'))
    model.eval()
    # 假设我们有一个新的图像数据
    image_path = './cifar-10/cifar_img/cifar/test/0_cat.png' # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()  # 将PIL图像转为Tensor并自动归一化到[0,1]
    ])

    image = transform(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f'Predicted: {predicted.item()}')