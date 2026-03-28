import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
            
        return x

class PANNs_CNN14(nn.Module):
    def __init__(self, num_classes=10):
        super(PANNs_CNN14, self).__init__()
        
        # We start directly at the Convolutional layers since your DataLoader 
        # already provides the Log-Mel Spectrograms.
        self.bn0 = nn.BatchNorm2d(64) # Normalize frequencies

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        # Final classification head (Replaced from 527 AudioSet classes to your 10 genres)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True) 

    def forward(self, x):
        # x shape: [Batch, 1, Mels, Time]
        
        # Align with PANNs native orientation: [Batch, 1, Time, Mels]
        x = x.transpose(2, 3)

        # Frequency-wise Batch Normalization
        # Swap 'Mels' (dim 3) into the channel slot (dim 1) for bn0
        x = x.transpose(1, 3) # Now: [Batch, Mels, Time, 1]
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Pass through the 6 VGG-style blocks
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Audio Pooling
        # Average pooling across time, max pooling across time, then sum them
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # Dense Layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Output probabilities
        output = self.fc_audioset(x)
        
        return output

def load_pretrained_cnn14(num_classes=10):
    """
    Downloads the official AudioSet weights, filters out the unused frontend layers,
    and maps the weights into our adapted CNN14 model.
    """
    model = PANNs_CNN14(num_classes=num_classes)
    
    # Official Zenodo link for CNN14 32kHz weights
    weight_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    weight_path = "Cnn14_mAP=0.431.pth"
    
    if not os.path.exists(weight_path):
        print("Downloading official PANNs CNN14 weights...")
        urllib.request.urlretrieve(weight_url, weight_path)
    
    print("Loading pretrained weights...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'spectrogram_extractor' in key or 'logmel_extractor' in key:
            continue
        if 'fc_audioset' in key:
            continue
        filtered_state_dict[key] = value
        
    # Load filtered weights (strict=False for missing keys)
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Pretrained CNN14 ready!")
    
    return model