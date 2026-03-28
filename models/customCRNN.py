import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes=10, n_mels=64):
        super().__init__()
        
        # BRANCH 1: Texture Extractor (CNN)
        self.cnn = nn.Sequential(
            self._conv_block(in_channels=1, out_channels=16),
            self._conv_block(in_channels=16, out_channels=32),
            self._conv_block(in_channels=32, out_channels=64),
            self._conv_block(in_channels=64, out_channels=128),
            nn.AdaptiveAvgPool2d((1, 1)) # Compress the remaining spatial dimensions to 1x1
        )

        # BRANCH 2: Rhythm Extractor (RNN (GRU))
        self.rnn = nn.GRU(
            input_size=n_mels, 
            hidden_size=64,
            num_layers=2, 
            batch_first=True, 
            bidirectional=True # Output will be 64 * 2 = 128 features
        )

        # THE FUSION & ATTENTION GATE
        # We will concatenate the 128 CNN features and 128 RNN features (Total 256).
        combined_features = 128 + 128
        self.attention_gate = nn.Sequential(
            nn.Linear(combined_features, combined_features),
            nn.Sigmoid()
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        # x shape: [Batch, Channels (1), Mels, Time]

        # BRANCH 1 (CNN)
        cnn_out = self.cnn(x) # Output: [Batch, 128, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1) # Flatten: [Batch, 128]

        # BRANCH 2 (RNN)
        # Squeeze the channel dimension (since it's 1)
        # Swap Mels and Time axes so Time is the sequence dimension for the RNN
        rnn_in = x.squeeze(1).transpose(1, 2) # Input: [Batch, Time, Mels]
        rnn_out, _ = self.rnn(rnn_in) # Output: [Batch, Time, 128]

        #take the last hidden representation
        rnn_out = rnn_out[:, -1, :] # Slice last step: [Batch, 128]

        # Stick the texture features and rhythm features together
        merged = torch.cat((cnn_out, rnn_out), dim=1) # [Batch, 256]
        
        # Generate importance weights and multiply them against the features
        attn_weights = self.attention_gate(merged) # [Batch, 256]
        attended_features = merged * attn_weights # [Batch, 256]

        # CLASSIFY
        output = self.classifier(attended_features) # [Batch, 10]
        
        return output