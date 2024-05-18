import torch
import torch.nn as nn
from typing import List, Tuple

class HorseshoeEncoder(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 architecture:List[Tuple]=[(2,16),(2,32),(3,64),(3,128),(3,128)],) -> None:
        super(HorseshoeEncoder, self).__init__()

        def conv_layer(in_channels, out_channels, *args, **kwargs):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        self.layers = nn.ModuleDict()

        in_channels = in_channels  # initial number of input channels
        for i, (n_blocks, out_channels) in enumerate(architecture):
            conv_layers = []
            for _ in range(n_blocks):
                conv_layers.append(conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=0))
                in_channels = out_channels
            conv_block = nn.Sequential(*conv_layers)
            # We add a MaxPooling layer after the prescribed number of conv layers
            self.layers.add_module(f'conv_block_{i}',nn.ModuleList([conv_block, nn.MaxPool2d(2,2,return_indices=True)]))

        self.conv1x1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        pooling_indices = []
        input_sizes = []
        #for block in self.layers.values():
        for i in range(len(self.layers)):
            conv_block, pooling_layer = self.layers[f'conv_block_{i}']
            # Apply the convolutional layer
            x = conv_block(x)
            # we need to keep track of the input size
            # before the pooling, to do the unpooling correctly at decoding time
            input_sizes.append(x.size())
            # Apply the pooling layer
            x, indices = pooling_layer(x)
            # We need to keep track of the indices for the unpooling
            pooling_indices.append(indices)
        x = self.conv1x1(x)

        return x, pooling_indices, input_sizes

class HorseshoeDecoder(nn.Module):
    def __init__(self,
                 out_channels:int=1,
                 architecture:List[Tuple]=[(2,16),(2,32),(3,64),(3,128),(3,128)],) -> None:
        super(HorseshoeDecoder, self).__init__()

        def transp_conv_layer(in_channels, out_channels, *args, **kwargs):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        self.layers = nn.ModuleDict()

        out_channels = out_channels     # desired n° of output channels
        arch_len = len(architecture)
        for i, (n_blocks, in_channels) in enumerate(architecture):
            transp_conv_layers = []
            for _ in range(n_blocks):
                transp_conv_layers.insert(0,transp_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=0))
                out_channels = in_channels
            transp_conv_block = nn.Sequential(*transp_conv_layers)
            # Since we're building the architecture from right to left, we need to add 
            # (more precisely, index) the modules in reversed order in the dictionary
            self.layers.add_module(f'transp_conv_block_{arch_len-i-1}',nn.ModuleList([nn.MaxUnpool2d(2,2), transp_conv_block])) 

        self.Tconv1x1 = nn.ConvTranspose2d(256, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, pooling_indices, input_sizes):
        x = self.Tconv1x1(x)
        #for block in reversed(self.layers.values()):
        for i in range(len(self.layers)):
            unpooling_layer, transp_conv_block = self.layers[f'transp_conv_block_{i}']
            # First the unpooling (which needs the indices used for pooling and the size of the input)...
            x = unpooling_layer(x, pooling_indices.pop(), output_size=input_sizes.pop())
            # ... then the transposed convolution!
            x = transp_conv_block(x)

        return x
    
class HorseshoeNetwork(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 architecture:List[Tuple]=[(2,16),(2,32),(3,64),(3,128),(3,128)],):
        super(HorseshoeNetwork, self).__init__()

        # Convolutional layers
        self.encoder = HorseshoeEncoder(architecture=architecture,in_channels=in_channels)

        # Transpose conv. layers
        self.decoder = HorseshoeDecoder(architecture=architecture,out_channels=out_channels)
        
    def forward(self, x):
        # Encoding
        x, pooling_indices, input_sizes = self.encoder(x)

        # Decoding
        x = self.decoder(x, pooling_indices, input_sizes)

        # Final sigmoid (classification)
        x = torch.sigmoid(x)
        return x


# Test with an input image tensor of shape [1, 3, 568, 800]
img = torch.randn(1, 3, 568, 800)

# Initialize the encoder and decoder with the specified architectures
encoder = HorseshoeEncoder(architecture=[(2, 16), (2, 32)])
decoder = HorseshoeDecoder(architecture=[(2, 16), (2, 32)])

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
