import os
import random
import json

def create_and_save_split(root_dir, save_path="dataset_split.json"):
    genres = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    split_data = {'train': {}, 'val': {}}
    
    for genre in genres:
        genre_path = os.path.join(root_dir, genre)
        songs = sorted([d for d in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, d))])
        
        # Shuffle to ensure random 85/15 selection
        random.shuffle(songs)
        
        train_songs = songs[:85]
        val_songs = songs[85:100]
        
        split_data['train'][genre] = ["/".join(os.path.join(genre_path, s).split("/")[-2:]) for s in train_songs]
        split_data['val'][genre] = ["/".join(os.path.join(genre_path, s).split("/")[-2:]) for s in val_songs]
        
    with open(save_path, 'w') as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Split created and saved to {save_path}")
    return split_data

split_data = create_and_save_split("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/genres_stems")