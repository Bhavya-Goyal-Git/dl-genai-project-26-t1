import torch
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    
    # tqdm progress bar for the Kaggle console
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Automatic Mixed Precision (AMP) for massive speedups
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # Backward pass with GradScaler (if AMP is enabled)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # Step the scheduler if it's a batch-level scheduler (like OneCycleLR)
        if scheduler is not None:
            scheduler.step()
            
        # Accumulate loss
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_train_loss = running_loss / len(dataloader)
    return avg_train_loss