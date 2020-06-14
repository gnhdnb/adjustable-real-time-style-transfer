import torch

class ResidualDense(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResidualDense, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_channels, in_channels, kernel_size = 1),
            torch.nn.ReLU()
        )
    
    def forward(self, X):
        return X - self.model(X)

class ConditionerNet(torch.nn.Module):
    def __init__(self):
        super(ConditionerNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 1000, kernel_size = 1),
            torch.nn.ReLU(),
            ResidualDense(1000, 1000),
            ResidualDense(1000, 1000),
            ResidualDense(1000, 1000),
            ResidualDense(1000, 1000),
            torch.nn.Conv2d(1000, 4224, kernel_size = 1)
        )
    
    def forward(self, X):
        return self.model(X)


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size = 9, stride = 1) 
        self.in1 = ContitionalInstanceNorm2d()

        self.conv2 = ConvLayer(32, 64, kernel_size = 3, stride = 2)
        self.in2 = ContitionalInstanceNorm2d()

        self.conv3 = ConvLayer(64, 128, kernel_size = 3, stride = 2)
        self.in3 = ContitionalInstanceNorm2d()

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)

        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size = 3, stride = 1, upsample = 2)
        self.in4 = ContitionalInstanceNorm2d()
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size = 3, stride = 1, upsample = 2)
        self.in5 = ContitionalInstanceNorm2d()
        self.deconv3 = ConvLayer(32, 3, kernel_size = 9, stride = 1)
        self.relu = torch.nn.ReLU()

    def forward(self, X, gammas, betas):
        y = self.relu(self.in1(self.conv1(X), gammas[:,0:32,:,:], betas[:,0:32,:,:]))
        y = self.relu(self.in2(self.conv2(y), gammas[:,32:96,:,:], betas[:,32:96,:,:]))
        y = self.relu(self.in3(self.conv3(y), gammas[:,96:224,:,:], betas[:,96:224,:,:]))
        y = self.res1(y, gammas[:,224:480,:,:], betas[:,224:480,:,:])
        y = self.res2(y, gammas[:,480:736,:,:], betas[:,480:736,:,:])
        y = self.res3(y, gammas[:,736:992,:,:], betas[:,736:992,:,:])
        y = self.res4(y, gammas[:,992:1248,:,:], betas[:,992:1248,:,:])
        y = self.res5(y, gammas[:,1248:1504,:,:], betas[:,1248:1504,:,:])
        y = self.res6(y, gammas[:,1600:1856,:,:], betas[:,1600:1856,:,:])
        y = self.res7(y, gammas[:,1856:2112,:,:], betas[:,1856:2112,:,:])
        y = self.relu(self.in4(self.deconv1(y), gammas[:,1504:1568,:,:], betas[:,1504:1568,:,:]))
        y = self.relu(self.in5(self.deconv2(y), gammas[:,1568:1600,:,:], betas[:,1568:1600,:,:]))
        y = self.deconv3(y) 
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = ContitionalInstanceNorm2d()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = ContitionalInstanceNorm2d()
        self.relu = torch.nn.ReLU()

    def forward(self, x, gammas, betas):
        residual = x
        out = self.relu(self.in1(self.conv1(x), gammas[:,0:self.channels,:,:], betas[:,0:self.channels,:,:]))
        out = self.in2(self.conv2(out), gammas[:,self.channels:self.channels * 2,:,:], betas[:,self.channels:self.channels * 2,:,:])
        out = out + residual # need relu right after
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class ContitionalInstanceNorm2d(torch.nn.Module):
    def __init__(self):
        super(ContitionalInstanceNorm2d, self).__init__()
    
    def forward(self, x, gammas, betas):
        std = x.std(dim = [2, 3], keepdim=True)
        mean = x.mean(dim = [2, 3], keepdim=True)

        l = (x - mean) / std

        return l * gammas + betas