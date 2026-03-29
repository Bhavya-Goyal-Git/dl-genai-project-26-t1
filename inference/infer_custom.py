import pandas as pd
import torch
from models.customCRNN import CRNN
from utility.testDataset import TestMashupDataset
from torch.utils.data import DataLoader
from singleInferenceCommon import generate_single_submission


GENRE_MAPPING = sorted(['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'])

test_df = pd.read_csv("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

custom_crnn_model = CRNN(num_classes=10, n_mels=128).to(device)
# Load best weights
checkpoint_custom = torch.load("models/custom_best.pth", map_location=device)
custom_crnn_model.load_state_dict(checkpoint_custom['model_state_dict'])

print("Best weights successfully loaded!")

test_dataset_custom_crnn = TestMashupDataset(
    test_df=test_df,
    music_dir="/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup",
    sample_rate=32000, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000
)

test_loader_crnn = DataLoader(
    test_dataset_custom_crnn,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

generate_single_submission(
   model=custom_crnn_model,
   dataloader=test_loader_crnn,
   device=device,
   genre_mapping=GENRE_MAPPING,
   output_csv="submission.csv"
)