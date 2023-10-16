import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from GAN_Model import CNN_Discriminator, CNN_Generator
from torchvision.utils import save_image

batch_size = 8
epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def deNormalize(x_hat):  # 把正态分布标准化的数据还原
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # mean的维度进行改变，原来：[3] => [3, 1, 1]; 实现在对应维度上所有的值进行对应操作
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

    x = x_hat * std + mean
    return x


img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])    [0.485,0.456,0.406]、image_std=[0.229,0.224,0.225]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

FaceDataset = datasets.ImageFolder('../data', transform=img_transform)  # 数据路径
dataloader = DataLoader(FaceDataset,
                        batch_size=batch_size,  # 批量大小
                        shuffle=False,  # 不要乱序
                        num_workers=1  # 多进程
                        )


def main():
    torch.manual_seed(23)

    G = CNN_Generator(512, 15 * 192 * 192).cuda()
    D = CNN_Discriminator().to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))
    criterion = nn.BCELoss().to(device)
    iter = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        label_xr = torch.ones([batch_size, 1], requires_grad=False).to(device)  # 生成标签
        label_xf = torch.zeros([batch_size, 1], requires_grad=False).to(device)  # 生成标签

        for i, (img, _) in enumerate(dataloader):
            iter += 1
            # 1.判别器训练五次，生成器训练一次
            img = img.to(device)
            for _ in range(5):
                # 1.1 真实数据
                out_xr = D(img)
                if img.shape[0] != batch_size:
                    label_xr = torch.ones([img.shape[0], 1], requires_grad=False).to(device)
                loss_xr = criterion(out_xr, label_xr)

                # 1.2 生成数据
                z = torch.randn(batch_size, 512).to(device)
                fake_img = G(z).detach()
                out_xf = D(fake_img)
                loss_xf = criterion(out_xf, label_xf)

                loss_D = loss_xr + loss_xf
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

                # print("1")

            # 2.判别器训练好了五次，接下来生成器训练一次
            z = torch.randn(batch_size, 512).to(device)
            fake_img_G = G(z)
            fake_img_D = D(fake_img_G)
            loss_G = criterion(fake_img_D, label_xf)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # if iter % 20 == 0:
            print('Epoch [{}/{}], Batch {},d_loss: {:.6f}, g_loss: {:.6f} '
            .format(
                epoch, epochs, i, loss_D.item(), loss_G.item(),
            ))
            fake_img = deNormalize(fake_img_G.cpu())
            save_image(fake_img, '../img/fake_images_{}.png'.format(iter))

    torch.save(G.state_dict(), '../pth/GAN_Model_generator.pth')
    torch.save(D.state_dict(), '../pth/GAN_Model_discriminator.pth')


if __name__ == "__main__":
    main()
