import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlock(nn.Module):
    '''
    Basic resnet block
    '''
    def __init__(self, in_channels, out_channels, kernel=3, debug=False):
        super(ResBlock, self).__init__()

        self.debug = debug

        #self.downsample = out_channels//in_channels
        self.downsample = 2 if out_channels > in_channels else 1
        #print('resnet block: kernel: %d, down: %d'%(kernel, down))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=self.downsample, padding=1)#, padding=(0,1) if self.downsample == 1 else 0)
        self.relu1 = nn.LeakyReLU() #nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, padding=1)# if self.downsample == 1 else 0)
        self.relu2 = nn.LeakyReLU() #nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):

        residual = x
        if self.debug: print('resblock, input:',x.size())

        out = self.conv1(x)
        if self.debug: print('resblock, after conv1:',out.size())
        out = self.relu1(out)
        out = self.conv2(out)
        if self.debug: print('resblock, after conv2:',out.size())

        if self.downsample > 1:
            residual = self.shortcut(x)

        if self.debug: print('resblock, shortcut:',residual.size())

        out += residual
        out = self.relu2(out)

        return out

class ResBlock_stride(nn.Module):
    '''
    Basic resnet block
    '''
    def __init__(self, in_channels, out_channels, down=3, kernel=7, debug=False):
        super(ResBlock_stride, self).__init__()

        self.debug = debug

        #self.downsample = out_channels//in_channels
        self.downsample = down if out_channels > in_channels else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=self.downsample, padding=3)#, padding=(0,1) if self.downsample == 1 else 0)
        self.relu1 = nn.LeakyReLU() #nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, padding=3)# if self.downsample == 1 else 0)
        self.relu2 = nn.LeakyReLU() #nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):

        residual = x
        if self.debug: print('resblock, input:',x.size())

        out = self.conv1(x)
        if self.debug: print('resblock, after conv1:',out.size())
        out = self.relu1(out)
        out = self.conv2(out)
        if self.debug: print('resblock, after conv2:',out.size())

        if self.downsample > 1:
            residual = self.shortcut(x)

        if self.debug: print('resblock, shortcut:',residual.size())

        out += residual
        out = self.relu2(out)

        return out

class ResNet_stride(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps, down=3, kernel=7, debug=False):
        super(ResNet_stride, self).__init__()

        self.debug = debug
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.down = down
        self.kernel = kernel

        #self.conv0 = nn.Conv1d(in_channels, fmaps[0], kernel_size=7, stride=1)#, padding=(0,1))
        self.conv0 = nn.Conv1d(in_channels, fmaps[0], kernel_size=7, stride=3)#, padding=(0,1))
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
        #self.layer8 = self.block_layers(1, [fmaps[3],fmaps[4]])
        #self.layer9 = self.block_layers(self.nblocks, [fmaps[4],fmaps[4]])

        self.GlobalMaxPool1d = nn.AdaptiveMaxPool1d((1,))
        self.fc = nn.Linear(self.fmaps[-1], 1)

    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            #layers.append(ResBlock(fmaps[0], fmaps[1], down=self.down, kernel=self.kernel, debug=self.debug))
            layers.append(ResBlock_stride(fmaps[0], fmaps[1], down=self.down, kernel=self.kernel, debug=self.debug))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.debug: print(x.size())
        # Run through ResNet
        x = self.conv0(x)
        if self.debug: print(x.size())
        x = F.leaky_relu(x)#F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        #if self.debug: print(x.size())

        x = self.layer1(x)
        if self.debug: print(x.size())
        x = self.layer2(x)
        if self.debug: print(x.size())
        x = self.layer3(x)
        if self.debug: print(x.size())
        x = self.layer4(x)
        if self.debug: print(x.size())
        x = self.layer5(x)
        if self.debug: print(x.size())
        x = self.layer6(x)
        if self.debug: print(x.size())
        x = self.layer7(x)
        #if self.debug: print(x.size())
        #x = self.layer8(x)
        #if self.debug: print(x.size())
        #x = self.layer9(x)
        if self.debug: print('pre-maxpool',x.size())

        x = self.GlobalMaxPool1d(x)
        if self.debug: print('post-maxpool',x.size())
        x = x.view(x.size()[0], self.fmaps[-1])
        if self.debug: print(x.size())
        # FC
        x = self.fc(x)
        if self.debug: print(x.size())

        return x

class ResNet_deep(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps, kernel=3, debug=False):
        super(ResNet_deep, self).__init__()

        self.debug = debug
        self.fmaps = fmaps
        self.nblocks = nblocks

        #self.conv0 = nn.Conv1d(in_channels, fmaps[0], kernel_size=7, stride=1)#, padding=(0,1))
        self.conv0 = nn.Conv1d(in_channels, fmaps[0], kernel_size=7, stride=2)#, padding=(0,1))
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]], kernel)
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]], kernel)
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]], kernel)
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]], kernel)
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]], kernel)
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]], kernel)
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]], kernel)
        self.layer8 = self.block_layers(1, [fmaps[3],fmaps[4]], kernel)
        self.layer9 = self.block_layers(self.nblocks, [fmaps[4],fmaps[4]], kernel)

        self.GlobalMaxPool1d = nn.AdaptiveMaxPool1d((1,))
        self.fc = nn.Linear(self.fmaps[-1], 1)

    def block_layers(self, nblocks, fmaps, kernel):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1], kernel, debug=self.debug))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.debug: print(x.size())
        # Run through ResNet
        x = self.conv0(x)
        if self.debug: print(x.size())
        x = F.leaky_relu(x)#F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        #if self.debug: print(x.size())

        x = self.layer1(x)
        if self.debug: print(x.size())
        x = self.layer2(x)
        if self.debug: print(x.size())
        x = self.layer3(x)
        if self.debug: print(x.size())
        x = self.layer4(x)
        if self.debug: print(x.size())
        x = self.layer5(x)
        if self.debug: print(x.size())
        x = self.layer6(x)
        if self.debug: print(x.size())
        x = self.layer7(x)
        if self.debug: print(x.size())
        x = self.layer8(x)
        if self.debug: print(x.size())
        x = self.layer9(x)
        if self.debug: print('pre-maxpool',x.size())

        x = self.GlobalMaxPool1d(x)
        if self.debug: print('post-maxpool',x.size())
        x = x.view(x.size()[0], self.fmaps[-1])
        if self.debug: print(x.size())
        # FC
        x = self.fc(x)
        if self.debug: print(x.size())

        return x

class ResNet_premax(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps, premax=64, kernel=3, debug=False):
        super(ResNet_premax, self).__init__()

        self.debug = debug
        self.premax = premax
        self.fmaps = fmaps
        self.nblocks = nblocks

        self.conv0 = nn.Conv1d(in_channels, fmaps[0], kernel_size=7, stride=1)#, padding=(0,1))
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]], kernel)
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]], kernel)
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]], kernel)
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]], kernel)
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]], kernel)

        self.GlobalMaxPool1d = nn.AdaptiveMaxPool1d((1,))
        self.fc = nn.Linear(self.fmaps[-1], 1)

    def block_layers(self, nblocks, fmaps, kernel):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1], kernel))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.debug: print(x.size())
        # Pre-maxpool
        # Reduce to 128 length
        x = F.max_pool1d(x, kernel_size=self.premax)#/3.
        if self.debug: print(x.size())
        # Run through ResNet
        x = self.conv0(x)
        if self.debug: print(x.size())
        x = F.leaky_relu(x)#F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2)
        #if self.debug: print(x.size())

        x = self.layer1(x)
        if self.debug: print(x.size())
        x = self.layer2(x)
        if self.debug: print(x.size())
        x = self.layer3(x)
        if self.debug: print(x.size())
        x = self.layer4(x)
        if self.debug: print(x.size())
        x = self.layer5(x)
        if self.debug: print('pre-maxpool',x.size())

        x = self.GlobalMaxPool1d(x)
        if self.debug: print('post-maxpool',x.size())
        x = x.view(x.size()[0], self.fmaps[-1])
        if self.debug: print(x.size())
        # FC
        x = self.fc(x)
        if self.debug: print(x.size())

        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
