import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

def prepare_audio_dataset(input_root: str, output_root: str, target_sr: int, duration_sec: float = 30.0, duration_flag=True):
    """
    Traverses a directory of .wav files, processes them (mono, resample, pad/crop), 
    and saves them as .pt tensors in a mirrored directory structure.
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Calculate the exact number of samples we need for the target duration
    target_samples = int(target_sr * duration_sec)
    
    # Find all .wav files in the input directory (recursively)
    wav_files = list(input_path.rglob("*.wav"))
    
    print(f"Found {len(wav_files)} .wav files. Processing at {target_sr} Hz...")
    
    for wav_file in tqdm(wav_files, desc=f"Converting to {target_sr}Hz .pt"):
        # 1. Mirror the directory structure
        relative_path = wav_file.relative_to(input_path)
        out_file = output_path / relative_path.with_suffix('.pt')
        
        # Create parent directories if they don't exist
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. Load the audio
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
            
        # 3. Convert to Mono (if stereo/multi-channel stems exist)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 4. Resample if the native SR doesn't match our target
        if sample_rate != target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        if duration_flag:
            # 5. Crop or Pad to strictly enforce the 30-second length
            current_samples = waveform.shape[1]
            
            if current_samples > target_samples:
                # Crop the audio (taking the first 30 seconds)
                waveform = waveform[:, :target_samples]
            elif current_samples < target_samples:
                # Pad with zeros (silence) at the end of the track
                padding_needed = target_samples - current_samples
                # F.pad format for 1D: (pad_left, pad_right)
                waveform = F.pad(waveform, (0, padding_needed))
            
        # 6. Save as a PyTorch tensor
        # We save as float32, which is standard for neural network inputs
        torch.save(waveform.to(torch.float32), out_file)

    print(f"Done! Dataset saved to {output_root}")


ORIGINAL_DATA_DIR = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/genres_stems"

# 1. Create the 32kHz dataset
prepare_audio_dataset(
    input_root=ORIGINAL_DATA_DIR, 
    output_root="/kaggle/working/tensors_32k", 
    target_sr=32000, 
    duration_sec=30.0
)

# 2. Create the 16kHz dataset
prepare_audio_dataset(
    input_root=ORIGINAL_DATA_DIR, 
    output_root="/kaggle/working/tensors_16k", 
    target_sr=16000, 
    duration_sec=30.0
)

# For environment noise
ORIGINAL_DATA_DIR = "/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/ESC-50-master/audio"

# 1. Create the 32kHz dataset
prepare_audio_dataset(
    input_root=ORIGINAL_DATA_DIR, 
    output_root="/kaggle/working/noisetensors_32k", 
    target_sr=32000, 
    duration_sec=30.0,
    duration_flag=False
)

# 2. Create the 16kHz dataset
prepare_audio_dataset(
    input_root=ORIGINAL_DATA_DIR, 
    output_root="/kaggle/working/noisetensors_16k", 
    target_sr=16000, 
    duration_sec=30.0,
    duration_flag=False
)