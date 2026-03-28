from commonTrainer import train_model
from models.ASTransformer import AST
from utility.mashupDataset import StemMashupDataset
from torch.utils.data import DataLoader
import json
import torch

NUM_EPOCHS = 30
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AST().to(device)
model_name = "ast"

with open("utility/dataset_split.json", "r") as f:
    split_dict = json.load(f)

train_dataset = StemMashupDataset(
    split_data=split_dict,
    split_type='train',
    music_dir="/kaggle/input/musicmashup32k/tensors_16k",
    noise_dir='/kaggle/input/datasets/bhavyagoyall/noise16k/noisetensors_16k',
    samples_per_genre=15000,
    sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, f_min=0, f_max=8000
)

val_dataset = StemMashupDataset(
    split_data=split_dict,
    split_type='val',
    music_dir="/kaggle/input/musicmashup32k/tensors_16k",
    noise_dir=None, # No noise for validation
    samples_per_genre=1500,
    sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, f_min=0, f_max=8000
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False, # No shuffle in validation
    num_workers=4,
    pin_memory=True
)

train_model(
    model=model,
    model_name=model_name,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    checkpoint_dir=CHECKPOINT_DIR
)