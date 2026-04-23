import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

# ✅ 1. 选择设备（自动）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. 网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 13 * 13, 128)
        self.relu = nn.ReLU()
        # self.silu = nn.SiLU()  # 可选
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.silu(x)  # 可选
        x = self.fc2(x)
        return x


def export_tensor_to_bin(tensor, filename):
    """导出为 float32 二进制"""
    np_array = tensor.detach().cpu().numpy().astype(np.float32).copy(order='C')
    np_array.tofile(filename)
    print(f"Exported {filename}, shape: {np_array.shape}")


def main():
    print("Loading MNIST dataset and training for 1 epoch...")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # ✅ pin_memory + num_workers 提升 GPU 速度
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ✅ 3. 模型放到 GPU
    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # ✅ 4. 数据搬到 GPU
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Training Progress: [{batch_idx}/{len(train_loader)}]")

    print("Training finished! Exporting weights...")

    os.makedirs('weights', exist_ok=True)

    export_tensor_to_bin(model.conv1.weight, 'weights/conv1_weight.bin')
    export_tensor_to_bin(model.conv1.bias, 'weights/conv1_bias.bin')

    export_tensor_to_bin(model.fc1.weight, 'weights/fc1_weight.bin')
    export_tensor_to_bin(model.fc1.bias, 'weights/fc1_bias.bin')

    export_tensor_to_bin(model.fc2.weight, 'weights/fc2_weight.bin')
    export_tensor_to_bin(model.fc2.bias, 'weights/fc2_bias.bin')

    # 测试图片
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_image, true_label = test_dataset[0]

    img_pil = transforms.ToPILImage()(test_image)
    img_pil.save('test_image.png')

    export_tensor_to_bin(test_image, 'test_image_pixels.bin')

    print(f"\nAll done! Label: {true_label}")


if __name__ == '__main__':
    main()