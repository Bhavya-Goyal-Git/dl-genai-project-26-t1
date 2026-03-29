import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def ensemble_and_submit(logits_cnn, logits_ast, song_ids, genre_mapping, ast_weight, output_csv):
    """
    Blends raw logits into probabilities, applies weights, and builds the Kaggle CSV.
    """
    cnn_weight = 1.0 - ast_weight
    
    # Briefly convert back to PyTorch tensors just to use the fast Softmax function
    t_logits_cnn = torch.tensor(logits_cnn)
    t_logits_ast = torch.tensor(logits_ast)
    
    probs_cnn = F.softmax(t_logits_cnn, dim=1).numpy()
    probs_ast = F.softmax(t_logits_ast, dim=1).numpy()
    
    # The mathematical blend
    blended_probs = (cnn_weight * probs_cnn) + (ast_weight * probs_ast)
    
    # Find the winning class indices
    pred_indices = np.argmax(blended_probs, axis=1)
    
    # Map back to string genres
    all_predicted_genres = [genre_mapping[idx] for idx in pred_indices]
    
    # Build the CSV
    submission_df = pd.DataFrame({
        'id': song_ids,
        'genre': all_predicted_genres
    })
    
    # Sort by ID for a clean Kaggle submission
    submission_df = submission_df.sort_values(by='id').reset_index(drop=True)
    submission_df.to_csv(output_csv, index=False)
    
    print(f"Saved {output_csv} (AST Weight: {ast_weight:.2f} | CNN14 Weight: {cnn_weight:.2f})")
    return submission_df