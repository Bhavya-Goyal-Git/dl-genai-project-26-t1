from get_logits import get_test_logits
import torch
import pandas as pd
from models.cnn14_PANN import PANNs_CNN14
from models.ASTransformer import AST
from utility.testDataset import TestMashupDataset
from torch.utils.data import DataLoader
from ensemblePredict import ensemble_and_submit

GENRE_MAPPING = sorted(['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'])
test_df = pd.read_csv("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/test.csv")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

AST_WEIGHT = 0.52

cnn14_model = PANNs_CNN14(num_classes=10).to(device)
# Load best weights
checkpoint_cnn14 = torch.load("models/cnn14_best.pth", map_location=device)
cnn14_model.load_state_dict(checkpoint_cnn14['model_state_dict'])

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

ids, cnn14_logits = get_test_logits(
   model=cnn14_model,
   dataloader=test_loader_cnn14,
   device=device,
)

ast_model = AST(num_classes=10).to(device)
# Load best weights
checkpoint_ast = torch.load("models/ast_best.pth", map_location=device)
ast_model.load_state_dict(checkpoint_ast['model_state_dict'])

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

ids, ast_logits =get_test_logits(
   model=ast_model,
   dataloader=test_loader_ast,
   device=device,
)

ensemble_and_submit(
    cnn14_logits, ast_logits, ids, GENRE_MAPPING, 
    ast_weight=AST_WEIGHT, 
    output_csv="submission.csv"
)