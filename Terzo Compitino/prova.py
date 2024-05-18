import torch
import torch.nn as nn
from typing import List, Tuple

class HorseshoeEncoder(nn.Module):
    def __init__(self, architecture: List[Tuple[int, int]] = [(1, 16), (1, 32)]) -> None:
        super(HorseshoeEncoder, self).__init__()

        def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        self.layers = nn.ModuleDict()
        in_channels = 3  # Initial number of input channels for the first convolution layer

        for i, (n_blocks, out_channels) in enumerate(architecture):
            conv_layers = []
            for _ in range(n_blocks):
                conv_layers.append(conv_layer(in_channels, out_channels))
                in_channels = out_channels
            conv_block = nn.Sequential(*conv_layers)
            self.layers.add_module(f'conv_block_{i}', nn.ModuleList([conv_block, nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]))

        self.conv1x1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        pooling_indices = []
        input_sizes = []

        for block in self.layers.values():
            conv_block, pool_layer = block
            x = conv_block(x)
            input_sizes.append(x.size())
            x, indices = pool_layer(x)
            pooling_indices.append(indices)
        
        x = self.conv1x1(x)

        return x, pooling_indices, input_sizes

class HorseshoeDecoder(nn.Module):
    def __init__(self, architecture: List[Tuple[int, int]] = [(1, 32), (1, 16)]) -> None:
        super(HorseshoeDecoder, self).__init__()

        def transp_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.layers = nn.ModuleDict()

        _, in_channels = architecture[0]

        self.Tconv1x1 = nn.ConvTranspose2d(256, in_channels, kernel_size=1, stride=1, padding=0)

        for i, (n_blocks, out_channels) in enumerate(architecture):
            transp_conv_layers = []
            for _ in range(n_blocks):
                transp_conv_layers.append(transp_conv_layer(in_channels, out_channels))
                in_channels = out_channels
            transp_conv_block = nn.Sequential(*transp_conv_layers)
            self.layers.add_module(f'transp_conv_block_{i}', nn.ModuleList([nn.MaxUnpool2d(kernel_size=2, stride=2), transp_conv_block]))

        self.final_conv1x1 = nn.ConvTranspose2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, pooling_indices, input_sizes):
        x = self.Tconv1x1(x)
        for block in self.layers.values():
            unpool_layer, transp_conv_block = block
            x = unpool_layer(x, pooling_indices.pop(), output_size=input_sizes.pop())
            x = transp_conv_block(x)
        x = self.final_conv1x1(x)
        return x

# Test with an input image tensor of shape [1, 3, 568, 800]
img = torch.randn(1, 3, 568, 800)

# Initialize the encoder and decoder with the specified architectures
encoder = HorseshoeEncoder(architecture=[(1, 16), (1, 32)])
decoder = HorseshoeDecoder(architecture=[(1, 32), (1, 16)])

# Move to device if necessary (for example, GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = img.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)

# Pass the input through the encoder
encoded_img, pooling_indices, input_sizes = encoder(img)

# Pass the encoded image through the decoder
decoded_img = decoder(encoded_img, pooling_indices, input_sizes)

# Check the shape of the output
print(f"Original shape: {img.shape}")
print(f"Encoded shape: {encoded_img.shape}")
print(f"Decoded shape: {decoded_img.shape}")
