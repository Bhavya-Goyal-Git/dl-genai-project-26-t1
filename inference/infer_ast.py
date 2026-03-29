import pandas as pd
import torch
from models.ASTransformer import AST
from utility.testDataset import TestMashupDataset
from torch.utils.data import DataLoader
from singleInferenceCommon import generate_single_submission

GENRE_MAPPING = sorted(['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'])

test_df = pd.read_csv("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ast_model = AST(num_classes=10).to(device)
# Load best weights
checkpoint_ast = torch.load("models/ast_best.pth", map_location=device)
ast_model.load_state_dict(checkpoint_ast['model_state_dict'])

print("Best weights successfully loaded!")

test_dataset_ast = TestMashupDataset(
    test_df=test_df,
    music_dir="/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup",
    sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, f_min=0, f_max=8000
)

test_loader_ast = DataLoader(
    test_dataset_ast,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

generate_single_submission(
   model=ast_model,
   dataloader=test_loader_ast,
   device=device,
   genre_mapping=GENRE_MAPPING,
   output_csv="submission.csv"
)