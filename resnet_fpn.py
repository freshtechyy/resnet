import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    # Scale factor of the number of output channels
    expansion = 4

    def __init__(self, in_channels, out_channels, 
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels*self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels*self.expansion,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels*self.expansion))
            

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels))


    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, n_blocks_list=[3, 4, 6, 3],
                 out_channels_list=[64, 128, 256, 512], num_channels=3):
        """
        Args:
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_block_lists: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, 
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1))

        # Create four convoluiontal layers
        in_channels = 64
        # For the first block of the second layer, do not downsample and use stride=1.
        self.conv2_x = self.CreateLayer(ResBlock, n_blocks_list[0], 
                                        in_channels, out_channels_list[0], stride=1)
        
        # For the first blocks of conv3_x - conv5_x layers, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(ResBlock, n_blocks_list[1], 
                                        out_channels_list[0]*ResBlock.expansion,
                                        out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(ResBlock, n_blocks_list[2],
                                        out_channels_list[1]*ResBlock.expansion,
                                        out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(ResBlock, n_blocks_list[3], 
                                        out_channels_list[2]*ResBlock.expansion,
                                        out_channels_list[3], stride=2)


    def forward(self, x):
        """
        Args: 
            x: input image
        Returns:
            C2: feature maps after conv2_x
            C3: feature maps after conv3_x
            C4: feature maps after conv4_x
            C5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        # Feature maps
        C2 = self.conv2_x(x)
        C3 = self.conv3_x(C2)
        C4 = self.conv4_x(C3)
        C5 = self.conv5_x(C4)

        return C2, C3, C4, C5


    def CreateLayer(self, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(ResBlock(in_channels, out_channels, 
                             stride=stride, is_first_block=True))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))

        return nn.Sequential(*layer)


class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, is_highest_block=False):
        """
        Args:
            in_channels: the number of input channels
            out_channels: the number of output channels
            is_highest_block: whether the block is at the highest level of pyramid
        """
        super().__init__()
        # 1x1 convolution to unify the number of feature map channels to 
        # a specified value for fusion (matching)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=1, padding=0)
        # apply 3x3 convolution for feature output
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1)
        self.is_highest_block = is_highest_block


    def forward(self, x, y):
        """
        Args: 
            x: input tensor at the current level
            y: input tesnor at the previous level (upper level)
        Returns:
            x: output tensor to the next level (lower level)
            out: output tensor at the curret level (after 1x1 and 3x3 convolutions)
        """
        x = self.conv1(x)
        if not self.is_highest_block:
            x += F.interpolate(y, scale_factor=2, mode="bilinear",
                               align_corners=True) 
        out = self.conv2(x)
        return x, out


class FPN(nn.Module):
    def __init__(self, expansion=4, in_channels_list=[64, 128, 256, 512], 
                 out_channels=256):
        """
        Args:
            expansion: expansion rate of ResBlock (1 for BasicBlock or 4 for BottleNeck)
            in_channels_list: list of the output channel numbers for conv2_x - conv5_x
            out_channels: target number of channels (256 by default)
        """
        super().__init__()

        # Create layers to generate P2-P6
        self.P2 = FPNBlock(in_channels_list[0]*expansion, out_channels=out_channels)
        self.P3 = FPNBlock(in_channels_list[1]*expansion, out_channels=out_channels)
        self.P4 = FPNBlock(in_channels_list[2]*expansion, out_channels=out_channels)
        self.P5 = FPNBlock(in_channels_list[3]*expansion, out_channels=out_channels, 
                           is_highest_block=True)
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)


    def forward(self, C2, C3, C4, C5):
        """
        Args:
            C2-C5: feature maps output by ResNet
        Returns:
            P2-P6: enchanced features by FPN
        """
        x, P5 = self.P5(C5, None)
        x, P4 = self.P4(C4, x)
        x, P3 = self.P3(C3, x)
        _, P2 = self.P2(C2, x)
        P6 = self.P6(P5)

        return P2, P3, P4, P5, P6


class ResnetFPN(nn.Module):
    def __init__(self, resnet_arch='50'):
        """
        Args:
            resnet_arch: the type of ResNet architecture
        """
        super().__init__()

        assert(resnet_arch in ['18', '34', '50', '101', '152'])

        ResBlock = BasicBlock if resnet_arch == '18' or resnet_arch == '34' else BottleNeck

        # Number of residual blocks for each conv layer (conv2_x - conv5_x)
        if resnet_arch == '18':
            n_blocks_list = [2, 2, 2, 2]
        elif resnet_arch == '34' or resnet_arch == '50':
            n_blocks_list = [3, 4, 6, 3]
        elif resnet_arch == '101':
            n_blocks_list = [3, 4, 23, 3]
        else:
            n_blocks_list = [3, 8, 36, 3]

        # Create ResNet
        self.resnet = ResNet(ResBlock, n_blocks_list=n_blocks_list)

        # Create FPN
        self.FPN = FPN(expansion=ResBlock.expansion)
        

    def forward(self, x):
        """
        Args:
            x: input tensor
        Returns:
            P2-P6: enhanced features
        """
        C2, C3, C4, C5 = self.resnet(x)
        P2, P3, P4, P5, P6 = self.FPN(C2, C3, C4, C5)
        return P2, P3, P4, P5, P6


if __name__ == "__main__":
    ### Resnet arch types
    #resnet_arch = '18'
    #resnet_arch = '34'
    resnet_arch = '50'
    #resnet_arch = '101'
    #resnet_arch = '152'

    ### Create ResNet+FPN
    net = ResnetFPN(resnet_arch)
    
    ### Input
    x = torch.randn((1, 3, 512, 800), dtype=torch.float32)

    ### Inference
    P2, P3, P4, P5, P6 = net(x)

    ### Output
    print(f"P2 shape is {P2.shape}")
    print(f"P3 shape is {P3.shape}")
    print(f"P4 shape is {P4.shape}")
    print(f"P5 shape is {P5.shape}")
    print(f"P6 shape is {P6.shape}")

    print("Done!")
