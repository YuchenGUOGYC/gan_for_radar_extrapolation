import os
import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from tensorboardX import SummaryWriter
import glob
import sys
sys.path.append('/data/jt/guoyuchen/opt/home/guoyuchen/radar_extrapolation_code ')
from models import SmaAt_UNet
from utils import check_dir, get_learning_rate
import arrow
import torch.optim as optim
from torchvision.models.vgg import vgg16
import torchvision.utils as vutils
import matplotlib.pyplot as plt



class weighted_mae_windows(nn.Module):
    def __init__(self, weights=(0.5, 1.2, 1.4, 1.6, 1.8, 2.),
                 thresholds=(5., 15., 30., 40., 45.)):
        super(weighted_mae_windows, self).__init__()
        assert len(thresholds) + 1 == len(weights)
        self.weights = weights
        self.threholds = thresholds

    def forward(self, predict, target):
        """
        :param input: nbatchs * nlengths * nheigths * nwidths
        :param target: nbatchs * nlengths * nheigths * nwidths
        :return:
        """
        balance_weights = torch.zeros_like(target)
        balance_weights[target < self.threholds[0]] = self.weights[0]
        for i, _ in enumerate(self.threholds[:-1]):
            balance_weights[(target >= self.threholds[i]) & (target < self.threholds[i + 1])] = self.weights[i + 1]
        balance_weights[target >= self.threholds[-1]] = self.weights[-1]
        mae = torch.mean(balance_weights * (torch.abs(predict - target)))
        return mae


class ConvBlock(nn.Module):
    def __init__(self,inChannals,outChannals,stride = 1):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(inChannals,outChannals,kernel_size=3,stride = stride,padding=1,padding_mode='reflect',bias=False)
        self.bn = nn.BatchNorm2d(outChannals)
        self.relu = nn.LeakyReLU()
        
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(20,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.relu1 = nn.LeakyReLU()
        
        self.convBlock1 = ConvBlock(64,64,stride = 2)
        self.convBlock2 = ConvBlock(64,128,stride = 1)
        self.convBlock3 = ConvBlock(128,128,stride = 2)
        self.convBlock4 = ConvBlock(128,256,stride = 1)
        self.convBlock5 = ConvBlock(256,256,stride = 2)
        self.convBlock6 = ConvBlock(256,512,stride = 1)
        self.convBlock7 = ConvBlock(512,512,stride = 2)
        
        self.avePool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(512,1024,kernel_size=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(1024,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.convBlock6(x)
        x = self.convBlock7(x)
        x = self.avePool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class getDataset(Dataset):#
    
    def __init__(self,refdir,width,length,dirsample):
        
        super(getDataset, self).__init__()
        
        self.refdir =  refdir
        
        self.width = width
        
        self.length = length
        
        self.n_samples = len(refdir)
        
        self.dirsample = dirsample
    
    
    def __getitem__(self,index):
        
        try:
            timesteps = list([np.array([fi.split('_')[-2][-12:] for fi in self.refdir])[index]])
            REFF = []
            for timestep in timesteps:

                time_sample = [arrow.get(timestep,'YYYYMMDDHHmm').shift(minutes  = int(6*i)).format('YYYYMMDDHHmm') for i in range(30)]


                REF = []

                for ss in time_sample:
                    if len(glob.glob(self.dirsample +ss+'*'+'.npz')) ==1:
                        file = glob.glob(self.dirsample +ss+'*'+'.npz')[0]
                        ref = np.load(file)['ref'][550:1050, 585:1085]
                        ref[np.where(np.isnan(ref) == True)] =0
                        
                        
                        
                        REF.append(ref) 
                    else:
                        break            
                REF = np.array(REF)

                if len(REF) == 30:

                    REFF.append(REF)
                    
            SAMPLE = np.array(REFF)

            SAMPLE_x = torch.from_numpy(SAMPLE[:,4:10,:,:]).to(torch.float32)
            #SAMPLE_x = torch.from_numpy(SAMPLE[:,6:10,:,:]).to(torch.float32)

            SAMPLE_y = torch.from_numpy(SAMPLE[:,10:,:,:]).to(torch.float32)
                
            #print(SAMPLE_x.shape,SAMPLE_y.shape,torch.from_numpy(SAMPLE).to(torch.float32).shape)
            return SAMPLE_x,SAMPLE_y,torch.from_numpy(SAMPLE).to(torch.float32)
        
        except:
            pass#####here is an error 
    
    
    def __len__(self):
        
        return self.n_samples



###########################train##################################

width = 500
length = 500
in_length = 6
out_length = 20
########### load generator
gan_model = SmaAt_UNet(in_length, out_length)
gan_model = torch.nn.DataParallel(gan_model, device_ids=[0,1,2,3,4,5,6]).cuda()
gan_model.load_state_dict(torch.load('model_BEIJING/'+'/netG_BJx10y20_15.pth'), strict=True)
########### load Discriminator
Dis = Discriminator()
Dis = torch.nn.DataParallel(Dis, device_ids=[0,1,2,3,4,5,6]).cuda()
Dis.load_state_dict(torch.load('model_BEIJING/'+'/netD_BJx10y20_15.pth'), strict=True)


netD_new = Dis
netG_new = gan_model

optimizerG = optim.Adam(netG_new.parameters())
optimizerD = optim.Adam(netD_new.parameters())

lossF = nn.SmoothL1Loss()
lossF2 = weighted_mae_windows()


vgg = vgg16(pretrained=True)#.to(device)
lossNetwork = torch.nn.DataParallel(nn.Sequential(*list(vgg.features)[:31]), device_ids=[0,1,2,3,4,5,6]).cuda().eval()
for param in lossNetwork.parameters():
    param.requires_grad = False  #让VGG停止学习

refdir =list(np.load('/data/jt/guoyuchen/list_30_npz.npz')['dir'])
dirsample = '/data/jt/data/guoyuchen/radar_sample/folder_I/2018/'###






EPOCHS = 100
bs = 10
train_sets = getDataset(refdir,width,length,dirsample)

train_loader = DataLoader(train_sets, batch_size=bs,num_workers=12,
                              pin_memory=True, shuffle=False, drop_last=True
                              )

weightgan = 0.1

weightVGG = 0.2


for epoch in range(EPOCHS)[16:]:
    #netD_new.train()
    #netG_new.train()
    processBar = tqdm(enumerate(train_loader,1))
    

    
    
    for i,(cropImg,sourceImg,Img) in processBar:
            cropImg = cropImg.view(cropImg.shape[0]*cropImg.shape[1],cropImg.shape[2],cropImg.shape[3],cropImg.shape[4]).cuda()
            sourceImg = sourceImg.view(sourceImg.shape[0]*sourceImg.shape[1],sourceImg.shape[2],sourceImg.shape[3],sourceImg.shape[4]).cuda()
            Img = Img.view(Img.shape[0]*Img.shape[1],Img.shape[2],Img.shape[3],Img.shape[4]).cuda()

            fakeImg = netG_new(cropImg)

            #迭代辨别器网络
            eppp = min([15,epoch])
            eppp = max([5, eppp])

            for i in range(eppp):
                netD_new.zero_grad()
                batchnum,channel,www,lll = sourceImg.size()
                realOut = netD_new(sourceImg).mean()
                fakeOut = netD_new(fakeImg).mean()
                dLoss = 1 - realOut + fakeOut
                dLoss.backward(retain_graph=True)

            
            #迭代生成器网络
            netG_new.zero_grad()
            
            gLossSR = lossF2(fakeImg,sourceImg) 
            
            gLossGAN = weightgan * torch.mean(1 - fakeOut)

            num1 = int(np.random.randint(8,27,(1,))[0])-4######
            num2 = int(num1+3)
            Fimg = torch.cat([cropImg,fakeImg],dim = 1)[:,num1:num2,:,:]
            Simg = Img[:,num1:num2,:,:]


            gLossVGG = weightVGG* lossF(lossNetwork(Fimg),lossNetwork(Simg))
            gLoss = gLossSR + gLossGAN + gLossVGG
            gLoss.backward()

            optimizerD.step()
            optimizerG.step()


            #数据可视化
            processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, EPOCHS, dLoss.item(),gLoss.item(),realOut.item(),fakeOut.item()))

    if realOut.item()<= fakeOut.item() and realOut.item() >= 10e-6:
        
        decay = 0.1
        
        weightgan = decay*weightgan
    
        weightVGG = decay*weightVGG
        print(weightgan,weightVGG)
    
    m_dir = 'model_BEIJING/'   
    check_dir(m_dir)
    #保存模型路径文件
    torch.save(netG_new.state_dict(), m_dir + '/netG_BJx10y20_%d.pth' % (epoch))
    torch.save(netD_new.state_dict(), m_dir + '/netD_BJx10y20_%d.pth' % (epoch))





