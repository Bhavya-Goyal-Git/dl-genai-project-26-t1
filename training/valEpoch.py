import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_f1_score

def validation_epoch(model, dataloader, criterion, device):
    """
    Performs one epoch of validation and calculates macro metrics.
    """
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    
    # Disable gradient tracking to save memory and compute
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # We still use autocast in eval for speed consistency
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            
            # Convert logits to class predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Move predictions and labels back to CPU for Scikit-Learn
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate average loss
    avg_val_loss = running_loss / len(dataloader)
    
    # Calculate standard accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate Macro Precision, Recall, and F1
    precision, recall, f1, _ = precision_recall_f1_score(
        all_labels, 
        all_preds, 
        average='macro', 
        zero_division=0
    )
    
    return avg_val_loss, accuracy, precision, recall, f1