import torch
from tqdm import tqdm
import pandas as pd

def generate_single_submission(model, dataloader, device, genre_mapping, output_csv="submission.csv"):
    """
    Runs inference on an unlabelled dataloader and generates submission CSV.
    """

    model.eval()
    all_song_ids = []
    all_predicted_genres = []
    
    # Disable gradient tracking for speed and memory
    with torch.no_grad():
        for inputs, song_ids in tqdm(dataloader, desc="Predicting Test Data"):
            inputs = inputs.to(device)
            
            # Use autocast to match how the model was trained
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                
            # Get the index of the highest probability
            pred_indices = torch.argmax(logits, dim=1)
            pred_indices = pred_indices.cpu().numpy()
            
            # If song_ids came back as a PyTorch tensor, convert them too
            if torch.is_tensor(song_ids):
                song_ids = song_ids.cpu().numpy()
                
            # 4. Map indices to string names and store them
            for idx, pred_idx in zip(song_ids, pred_indices):
                all_song_ids.append(idx)
                all_predicted_genres.append(genre_mapping[pred_idx])
                
    # Build submission.csv
    submission_df = pd.DataFrame({
        'id': all_song_ids,
        'genre': all_predicted_genres
    })
    submission_df.to_csv(output_csv, index=False)
    
    print(f"Successfully generated {output_csv} with {len(submission_df)} predictions!")
    
    return submission_df