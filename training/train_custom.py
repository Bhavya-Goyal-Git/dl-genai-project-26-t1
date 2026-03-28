from commonTrainer import train_model
from models.customCRNN import CRNN
from utility.mashupDataset import StemMashupDataset
from torch.utils.data import DataLoader
import json
import torch

NUM_EPOCHS = 200
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN().to(device)
model_name = "custom"

with open("utility/dataset_split.json", "r") as f:
    split_dict = json.load(f)

train_dataset = StemMashupDataset(
    split_data=split_dict, 
    split_type='train',
    music_dir="/kaggle/input/musicmashup32k/tensors_32k",
    noise_dir='/kaggle/input/datasets/bhavyagoyall/noise32k/noisetensors_32k', 
    samples_per_genre=15000,
    sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000
)

val_dataset = StemMashupDataset(
    split_data=split_dict, 
    split_type='val',
    music_dir="/kaggle/input/musicmashup32k/tensors_32k",
    noise_dir=None, # No noise for validation
    samples_per_genre=1500,
    sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
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