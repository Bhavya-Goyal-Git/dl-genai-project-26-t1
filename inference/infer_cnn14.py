import pandas as pd
import torch
from models.cnn14_PANN import PANNs_CNN14
from utility.testDataset import TestMashupDataset
from torch.utils.data import DataLoader
from singleInferenceCommon import generate_single_submission

GENRE_MAPPING = sorted(['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'])

test_df = pd.read_csv("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn14_model = PANNs_CNN14(num_classes=10).to(device)
# Load best weights
checkpoint_cnn14 = torch.load("models/cnn14_best.pth", map_location=device)
cnn14_model.load_state_dict(checkpoint_cnn14['model_state_dict'])

print("Best weights successfully loaded!")

test_dataset_cnn14 = TestMashupDataset(
    test_df=test_df,
    music_dir="/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup",
    sample_rate=32000, n_fft=1024, hop_length=320, n_mels=64, f_min=50, f_max=14000
)

test_loader_cnn14 = DataLoader(
    test_dataset_cnn14,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

generate_single_submission(
   model=cnn14_model,
   dataloader=test_loader_cnn14,
   device=device,
   genre_mapping=GENRE_MAPPING,
   output_csv="submission.csv"
)