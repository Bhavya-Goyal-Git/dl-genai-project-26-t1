import torch
import torch.nn as nn
import random
from transformers import ASTForAudioClassification

class AST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Official MIT pretrained AST model from Hugging Face
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # AST expects exactly 1024 frames (10.24 seconds at 16kHz/160hop)
        self.target_frames = 1024

    def forward(self, x):
        # Input shape from DataLoader: [Batch, 1, Mels (128), Time (~3000)]
        batch_size = x.size(0)
        total_frames = x.size(-1)
        
        # AST doesn't need the channel dimension
        x = x.squeeze(1) # Now: [Batch, 128, Time]
        
        if self.training:
            # TRAINING: Random 10.24s Crop
            
            # Pick a random starting frame
            max_start = max(0, total_frames - self.target_frames)
            start_idx = random.randint(0, max_start)
            
            # Slice the chunk
            chunk = x[:, :, start_idx : start_idx + self.target_frames]
            
            # Hugging Face AST expects inputs as: [Batch, Time, Mels]
            chunk = chunk.transpose(1, 2) 
            
            # Get logits
            output = self.ast(chunk).logits
            return output
            
        else:
            # INFERENCE/VAL: 3-Crop Ensembling
            # We take 3 chunks to cover the 30 seconds: 
            starts = [0, 988, 1976] # 0-10s, 10-20s, and 20-30s (approximate indices)
            
            chunks = []
            for start_idx in starts:
                # Ensure we don't go out of bounds if the track is slightly short
                if start_idx + self.target_frames > total_frames:
                    start_idx = total_frames - self.target_frames
                    
                chunk = x[:, :, start_idx : start_idx + self.target_frames]
                chunk = chunk.transpose(1, 2) # [Batch, Time, Mels]
                chunks.append(chunk)
                
            # Stack all chunks together to process them in one massive batch
            # Shape becomes: [Batch * 3, 1024, 128]
            stacked_chunks = torch.cat(chunks, dim=0)
            
            # Pass all crops through the model simultaneously
            stacked_logits = self.ast(stacked_chunks).logits # [Batch * 3, 10]
            
            # Reshape back to [3, Batch, 10] and calculate the mean across the 3 crops
            # This averages the model's opinion of the Start, Middle, and End.
            averaged_logits = stacked_logits.view(3, batch_size, -1).mean(dim=0)
            
            return averaged_logits