import torch.nn as nn
import torch

class CNN_Generator(nn.Module): # 模型的输入为潜在空间的噪声维度，【batchsize，z_dimension】
    def __init__(self,z_dimension,num_feature):
        super(CNN_Generator, self).__init__()
        self.fc = nn.Linear(z_dimension, num_feature)  # batch, 15*192*192
        #第一层全连接，首先将噪声和标签信息拼接，映射成一个高维的向量
        self.br = nn.Sequential(
            nn.BatchNorm2d(15),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(15, 50, 3, stride=1, padding=1),  # batch, 50, 192, 192
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 192, 192
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 2, stride=2),  # batch, 3, 96, 96
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 15, 192, 192)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


class CNN_Discriminator(nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()

        # self.label_embedding = nn.Linear(label_dim, 96*96)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  # batch, 32, 96，96,
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 48, 48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 48, 48
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=3)  # batch, 64, 16, 16
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=3

        '''
        # label = self.label_embedding(label).view(-1,1,96,96)
        # x = torch.cat([x,label],1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



if __name__ == "__main__":
    z = torch.randn(8, 512)
    print(z.shape)
    G = CNN_Generator(512, 15*192*192)
    b = G(z)
    print(b.shape)

    D = CNN_Discriminator()
    d = D(b)
    print(d.shape)