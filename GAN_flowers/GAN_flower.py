import torch
from torch import nn,optim
from torch.utils.data import Dataset,TensorDataset,DataLoader
import tqdm

import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import time
import os
import subprocess

script_path = os.path.dirname(os.path.abspath(__file__))
sh_path = os.path.join(script_path,"downloader.sh")
print(script_path,sh_path)
subprocess.run(["bash",sh_path])


img_data=ImageFolder("./oxford-102/",
                     transform=transforms.Compose([transforms.Resize(80),transforms.CenterCrop(64),transforms.ToTensor()]))
batch_size=64
img_loader=DataLoader(img_data,batch_size=batch_size,shuffle=True)



nz=100
ngf=32

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2,ngf*1,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*1,3,4,2,1,bias=True),
            nn.Tanh()
            )

    def forward(self,x):
        out=self.main(x)
        return out


ndf=32

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*8,1,4,1,0,bias=True)
            )

    def forward(self,x):
        out=self.main(x)
        return out.squeeze()


d=DNet().to("cuda:0")
g=GNet().to("cuda:0")

opt_d=optim.Adam(d.parameters(),lr=0.0002,betas=(0.5,0.999))
opt_g=optim.Adam(g.parameters(),lr=0.0002,betas=(0.5,0.999))

ones=torch.ones(batch_size).to("cuda:0")
zeros=torch.zeros(batch_size).to("cuda:0")

loss_f=nn.BCEWithLogitsLoss()

fixed_z=torch.randn(batch_size,nz,1,1).to("cuda:0")



from statistics import mean

def train_dcgan(g,d,opt_g,opt_d,loader):
    log_loss_g=[]
    log_loss_d=[]
    for real_img,_ in tqdm.tqdm(loader):
        batch_len=len(real_img)
        real_img=real_img.to("cuda:0")
        z=torch.randn(batch_len,nz,1,1).to("cuda:0")
        #fake_img=g.forward(z)
        fake_img=g(z)
        fake_img_tensor=fake_img.detach()
        out=d.forward(fake_img)
        loss_g=loss_f(out,ones[:batch_len])
        log_loss_g.append(loss_g.item())
        d.zero_grad(),g.zero_grad()
        loss_g.backward()
        opt_g.step()
        real_out=d.forward(real_img)
        loss_d_real=loss_f(real_out,ones[:batch_len])

        fake_img=fake_img_tensor
        fake_out=d.forward(fake_img_tensor)
        loss_d_fake=loss_f(fake_out,zeros[:batch_len])

        loss_d=loss_d_real+loss_d_fake
        log_loss_d.append(loss_d.item())
        d.zero_grad(),g.zero_grad()
        loss_d.backward()
        opt_d.step()

    return mean(log_loss_g),mean(log_loss_d)


if not os.path.exists("data"):
    os.mkdir("data")


from PIL import Image
import glob

total_time=0.0
average_time=0.0
iter_epoch=100

gif_data = []

for epoch in range(iter_epoch):
    t_start=time.time()
    print("current epoch : ",epoch)
    train_dcgan(g,d,opt_g,opt_d,img_loader)
    t_finish=time.time()
    total_time+=t_finish-t_start
    average_time=total_time/(epoch+1)
    print("runtime = ",t_finish-t_start,"sec")
    print("expected remaining time = ",int(average_time*(iter_epoch-(epoch+1))),"sec",":",int(average_time*(iter_epoch-(epoch+1))/60),"minute")
    gif_data.append(g(gif_z))
    if epoch%1==0:
        torch.save(g.state_dict(),"./data/g_{:03d}.prm".format(epoch),pickle_protocol=4)
        torch.save(d.state_dict(),"./data/d_{:03d}.prm".format(epoch),pickle_protocol=4)
        generated_img=g(fixed_z)
        save_image(generated_img,"./data/{:03d}.jpg".format(epoch))
        


files = sorted(glob.glob('./data/*.jpg'))  
images = list(map(lambda file : Image.open(file) , files))
images[0].save('generating_process.gif' , save_all = True , append_images = images[1:] , duration = 200 , loop = 0)



