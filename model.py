import torch
import torch.nn as nn


# Downsampling block
class DownSample(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    
# Upsampling block
class UpSample(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Max pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Downsampling
        self.down_block_1 = DownSample(3, 64)
        self.down_block_2 = DownSample(64, 128)
        self.down_block_3 = DownSample(128, 256)
        self.down_block_4 = DownSample(256, 512)

        #Bridge
        self.bridge = UpSample(512, 1024) # Data from this is not concated to other layers

        #Upsampling
        self.conv_trans_1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2)
        self.up_block_4 = UpSample(1024,512)
        self.conv_trans_2 = nn.ConvTranspose2d(512,512, kernel_size=2, stride=2)
        self.up_block_3 = UpSample(768,256)
        self.conv_trans_3 = nn.ConvTranspose2d(256,256, kernel_size=2, stride=2)
        self.up_block_2 = UpSample(384,128)
        self.conv_trans_4 = nn.ConvTranspose2d(128,128, kernel_size=2, stride=2)
        self.up_block_1 = UpSample(192,64)

        #Final output
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=3, stride=1, kernel_size=3, padding=1)
        self.act = nn.Tanh()

    def forward(self, x):
        
        #Downsample portion
        e1 = self.down_block_1(x)
        x1 = self.max_pool(e1)
        e2 = self.down_block_2(x1)
        x2= self.max_pool(e2)
        e3 = self.down_block_3(x2)
        x3 = self.max_pool(e3)
        e4 = self.down_block_4(x3)
        x4 = self.max_pool(e4)

        #Bridge
        e5 = self.bridge(x4)
        x5 = self.conv_trans_1(e5)

        #Upsampling portion
        e6 = self.up_block_4(torch.cat([e4,x5],dim=1))
        x6 = self.conv_trans_2(e6)
        e7 = self.up_block_3(torch.cat([e3,x6], dim=1))
        x7 = self.conv_trans_3(e7)
        e8 = self.up_block_2(torch.cat([e2,x7],dim=1))
        x8 = self.conv_trans_4(e8)
        e9 = self.up_block_1(torch.cat([e1,x8],dim=1))

        #Final layer
        x10 = self.final_layer(e9)
        return self.act(x10)