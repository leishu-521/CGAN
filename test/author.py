import torch.nn as nn
import torch


class CNN_Generator(nn.Module):
    def __init__(self, noize_dim, label_dim, num_feature):
        super(CNN_Generator, self).__init__()
        # 第一层全连接，首先将噪声和标签信息拼接，映射成一个高维的向量
        self.fc = nn.Linear(noize_dim + label_dim, num_feature)
        # batch, 15*192*192
        self.br = nn.Sequential(
            nn.BatchNorm2d(15),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(15, 50, 3, stride=1, padding=1),
            # batch, 50, 192, 192             nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            # batch, 25, 192, 192             nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 2, stride=2),
            # batch, 3, 96, 96             nn.Tanh()
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
    # 判别器的思路：首先把标签通过全连接层  变换成一个矩阵，作为原图的一个通道，拼接过去。 然后对整个图片变成一个概率值
    def __init__(self, label_dim):
        super(CNN_Discriminator, self).__init__()

        self.label_embedding = nn.Linear(label_dim, 96 * 96)  # 全连接层
        self.conv1 = nn.Sequential(
            # batch, 32, 96，96,             nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, padding=2),
            # batch, 32, 48, 48
            nn.AvgPool2d(2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 48, 48             nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=3))  # batch, 64, 16, 16
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        ''' x: batch, width, height, channel=3 '''
        label = self.label_embedding(label).view(-1, 1, 96, 96)
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):

        if (i + 1) * batch_size < num_samples:
            batch_vectors = torch.cat((label_vectors[i * batch_size:(i + 1) * batch_size]), 0)
            batch_vectors = batch_vectors.view(-1, 32).float()
        else:
            batch_vectors = torch.cat((label_vectors[i * batch_size:-1]), 0)
            batch_vectors = batch_vectors.view(-1, 32).float()
        # 此处label_vectors为存储了所有真实图片标签onehot编码向量的列表 		#batch_vectors是为了选出和这一个batch匹配的label向量

        num_img = img.size(0)
        # train discriminator 		# compute loss of real_matched_img 		img = img.view(num_img,3,96,96)
        real_img = Variable(img).to(device)
        real_label = Variable(torch.ones(num_img)).to(device)
        fake_label = Variable(torch.zeros(num_img)).to(device)
        batch_vectors = Variable(batch_vectors).to(device)
        matched_real_out = D(img, batch_vectors)
        d_loss_matched_real = criterion(matched_real_out, real_label)
        matched_real_scores = matched_real_out  # closer to 1 means better
        # compute loss of fake_matched_img 		z = Variable(torch.randn(num_img, z_dimension)).to(device)
        z = torch.cat((z, batch_vectors), axis=1).to(device)
        fake_img = G(z)
        matched_fake_out = D(fake_img, batch_vectors)
        d_loss_matched_fake = criterion(matched_fake_out, fake_label)
        matched_fake_out_scores = matched_fake_out  # closer to 0 means better
        # compute loss of real_unmatched_img 		rand_label_vectors=random.sample(label_vectors,num_img)
        rand_batch_vectors = torch.cat((rand_label_vectors[:]), 0)
        rand_batch_vectors = rand_batch_vectors.view(-1, 32).float()
        # 错误向量用随机选取的方式选取，由于本数据集中经过预处理后相同的图片描述很少，所以采用这一种方法

        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        z = torch.cat((z, rand_batch_vectors), axis=1).to(device)
        fake_img = G(z)
        unmatched_real_out = D(fake_img, batch_vectors)
        d_loss_unmatched_real = criterion(unmatched_real_out, fake_label)
        unmatched_real_out_scores = unmatched_real_out  # closer to 0 means better
        # bp and optimize 		d_loss = d_loss_matched_real + d_loss_matched_fake + d_loss_unmatched_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator 		# compute loss of fake_img 		# compute loss of fake_matched_img 		z = Variable(torch.randn(num_img, z_dimension)).to(device)
        z = torch.cat((z, batch_vectors), axis=1).to(device)
        fake_img = G(z)
        matched_fake_out = D(fake_img, batch_vectors)
        matched_fake_out_scores = matched_fake_out

        g_loss = criterion(matched_fake_out, real_label)

        # bp and optimize 		g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
