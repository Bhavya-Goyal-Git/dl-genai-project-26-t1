import torch
import torchaudio
import torch.nn.functional as F
import gradio as gr
from models.cnn14_PANN import PANNs_CNN14
from models.ASTransformer import AST
from huggingface_hub import hf_hub_download
import os

GENRE_MAPPING = sorted(['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'])

class EnsemblePredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize models
        self.model_cnn14 = PANNs_CNN14(num_classes=10).to(self.device)
        self.model_ast = AST(num_classes=10).to(self.device)

        repo_id = "Bhavya-Goyal-Hf/dl-genai-weights" 
        token = os.environ.get("HF_TOKEN")

        cnn_path = hf_hub_download(repo_id=repo_id, filename="cnn14_best.pth", token=token)
        ast_path = hf_hub_download(repo_id=repo_id, filename="ast_best.pth", token=token)
        
        # Load the weights
        # Note: In HF Spaces, the working directory is the root of your repository
        cnn_ckpt = torch.load(cnn_path, map_location=self.device)
        ast_ckpt = torch.load(ast_path, map_location=self.device)
        
        self.model_cnn14.load_state_dict(cnn_ckpt['model_state_dict'])
        self.model_ast.load_state_dict(ast_ckpt['model_state_dict'])
        
        self.model_cnn14.eval()
        self.model_ast.eval()
        
    def preprocess_audio(self, filepath, sr, n_fft, hop, mels, f_min, f_max, win=None):
        """Dynamic audio processor for both 32kHz and 16kHz pipelines."""
        waveform, orig_sr = torchaudio.load(filepath)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=sr)
            
        # Strict 30-Second Trimming/Padding
        target_samples = int(sr * 30.0)
        current_samples = waveform.shape[1]
        if current_samples > target_samples:
            waveform = waveform[:, :target_samples]
        elif current_samples < target_samples:
            waveform = F.pad(waveform, (0, target_samples - current_samples))
            
        # Peak Normalization
        abs_max = waveform.abs().max()
        if abs_max > 0:
            waveform = waveform / abs_max
            
        # Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win if win else n_fft,
            hop_length=hop, n_mels=mels, f_min=f_min, f_max=f_max
        )
        amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        
        log_mel_spec = amp_to_db(mel_transform(waveform))
        
        # Instance Normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-5)
        
        # Add Batch Dimension [1, Channels, Mels, Time]
        return log_mel_spec.unsqueeze(0).to(self.device)

    def predict(self, filepath, ast_weight):
        """The main Gradio inference function."""
    
        if filepath is None:
            return "Please upload an audio file."
            
        cnn_weight = 1.0 - ast_weight
        
        # Process for CNN14 (32kHz)
        tensor_cnn = self.preprocess_audio(
            filepath, sr=32000, n_fft=1024, hop=320, mels=64, f_min=50, f_max=14000
        )
        
        # Process for AST (16kHz, win_length=400)
        tensor_ast = self.preprocess_audio(
            filepath, sr=16000, n_fft=1024, hop=160, mels=128, f_min=0, f_max=8000, win=400
        )
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_cnn = self.model_cnn14(tensor_cnn)
                logits_ast = self.model_ast(tensor_ast)
                
        # Softmax to Probabilities
        probs_cnn = F.softmax(logits_cnn, dim=1).squeeze()
        probs_ast = F.softmax(logits_ast, dim=1).squeeze()
        
        # The Mathematical Blend
        blended_probs = (cnn_weight * probs_cnn) + (ast_weight * probs_ast)
        
        # Format for Gradio's Label Component (Dictionary of {Genre: Probability})
        confidences = {GENRE_MAPPING[i]: float(blended_probs[i]) for i in range(10)}
        
        return confidences

# Instantiate the predictor once so models stay in RAM
predictor = EnsemblePredictor()


# Gradio Interface Setup
interface = gr.Interface(
    fn=predictor.predict,
    inputs=[
        gr.Audio(type="filepath", label="Upload Mashup (.wav)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.52, step=0.01, label="AST Model Weight (0 = Only CNN14, 1 = Only AST)")
    ],
    outputs=gr.Label(num_top_classes=4, label="Predicted Genre & Confidence"),
    title="Mashup Genre Classifier (DLGenAi Project)",
    description="Upload a 30-second audio clip to classify its genre. Adjust the slider to test different ensemble weights between the CNN14 and AST architectures in real-time!",
    theme="huggingface"
)

if __name__ == "__main__":
    interface.launch()