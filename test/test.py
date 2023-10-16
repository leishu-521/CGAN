from torchvision import datasets,transforms
import torch
import sys
# import

# batch_size = 64
# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0.5], [0.5])    [0.485,0.456,0.406]、image_std=[0.229,0.224,0.225]
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# FaceDataset = datasets.ImageFolder('../data', transform=img_transform)  # 数据路径
# dataloader = torch.utils.data.DataLoader(FaceDataset,
#                                          batch_size=batch_size,  # 批量大小
#                                          shuffle=False,  # 不要乱序
#                                          num_workers=1  # 多进程
#                                          )
#
#
# for i, (img,_) in enumerate(dataloader):
#     print(img.shape)
#     # print(type(_))
#     print("1")

# label_xr = torch.ones(2, requires_grad=False) # 生成标签
# print(label_xr.shape)

img = torch.randn([5,3,96,96])

# label_xr = torch.ones(10, requires_grad=False)  # 生成标签
# label_xr = torch.ones(img.shape[0], requires_grad=False)
# print(label_xr.shape)

def deNormalize(x_hat):  # 把正态分布标准化的数据还原
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # mean的维度进行改变，原来：[3] => [3, 1, 1]; 实现在对应维度上所有的值进行对应操作
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

    x = x_hat * std + mean
    return x

b = deNormalize(img)