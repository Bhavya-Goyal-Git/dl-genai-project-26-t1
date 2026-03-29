import torch
from tqdm import tqdm
import numpy as np

def get_test_logits(model, dataloader, device):
    """
    Runs inference and returns the raw logits and song IDs as numpy arrays.
    """
    model.eval()
    all_song_ids = []
    all_logits = []
    
    with torch.no_grad():
        for inputs, song_ids in tqdm(dataloader, desc="Extracting Test Logits"):
            inputs = inputs.to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                
            # Move logits to CPU and store
            all_logits.append(logits.cpu().numpy())
            
            if torch.is_tensor(song_ids):
                song_ids = song_ids.cpu().numpy()
                
            all_song_ids.extend(song_ids)
            
    # Concatenate into massive numpy arrays
    final_ids = np.array(all_song_ids)
    final_logits = np.concatenate(all_logits, axis=0)
    
    return final_ids, final_logits