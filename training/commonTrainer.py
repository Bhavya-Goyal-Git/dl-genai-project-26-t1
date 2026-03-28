import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from trainEpoch import train_epoch
from valEpoch import validation_epoch
import wandb

def train_model(
    model, 
    model_name, # 'custom', 'cnn14', or 'ast'
    train_loader, 
    val_loader, 
    device, 
    num_epochs=30, 
    checkpoint_dir="./checkpoints",
    resume_checkpoint=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    

    # 1. OPTIMIZER & DIFFERENTIAL LEARNING RATES
    if model_name == 'custom':
        # The custom model learns everything from scratch at a uniform rate
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        # Pre-trained models: Isolate the classification head from the backbone
        head_params = []
        backbone_params = []
        
        for name, param in model.named_parameters():
            # 'fc_audioset' matches our CNN14 head, 'classifier' matches the HuggingFace AST head
            if 'fc_audioset' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        # Backbone gets a gentle 5e-5, the new head gets a standard 1e-3
        optimizer = AdamW([
            {'params': backbone_params, 'lr': 5e-5},
            {'params': head_params, 'lr': 1e-3}
        ], weight_decay=1e-4)

    # 2. SCHEDULERS
    batch_scheduler = None
    epoch_scheduler = None
    
    if model_name == 'custom':
        # Epoch-level scheduler: Drops LR by half if Val Loss stalls for 3 epochs
        epoch_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        # Batch-level scheduler: OneCycleLR needs the max_lr for both parameter groups
        batch_scheduler = OneCycleLR(
            optimizer, 
            max_lr=[5e-5, 1e-3], 
            steps_per_epoch=len(train_loader), 
            epochs=num_epochs,
            pct_start=0.1 # Spend the first 10% of training warming up
        )


    # 3. LOSS & MIXED PRECISION
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')


    # 4. CHECKPOINT RESUMPTION
    start_epoch = 0
    best_val_f1 = 0.0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if model_name == 'custom':
            epoch_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            batch_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed at epoch {start_epoch} with Best F1: {best_val_f1:.4f}")


    # 5. THE MAIN TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} [{model_name.upper()}]")
        
        # Pass the batch_scheduler (used by OneCycleLR)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, batch_scheduler)
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = validation_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1,
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Step the epoch_scheduler (used by ReduceLROnPlateau)
        if epoch_scheduler is not None:
            epoch_scheduler.step(val_loss)


        # 6. SAVING LOGIC (Checkpointing)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': epoch_scheduler.state_dict() if model_name == 'custom' else batch_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_f1': best_val_f1,
            'val_loss': val_loss
        }
        
        # Save the latest epoch automatically (resume if Kaggle disconnects)
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"{model_name}_latest.pth"))
        
        # Save the Best Model based on Macro F1 Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"*** New Best F1 Score: {best_val_f1:.4f}! Saving model... ***")
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"{model_name}_best.pth"))

    print(f"\nTraining Complete! Best {model_name.upper()} Validation F1: {best_val_f1:.4f}")

    wandb.finish()
    
    return model