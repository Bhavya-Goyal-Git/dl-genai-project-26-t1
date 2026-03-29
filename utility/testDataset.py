from torch.utils.data import Dataset
import torchaudio
import torch
import os
import torch.nn.functional as F

class TestMashupDataset(Dataset):
    def __init__(self, test_df, sample_rate=32000, duration_sec=30.0, music_dir = None,
                 n_fft=1024, hop_length=320, n_mels=64, 
                 f_min=50, f_max=14000, win_length=None):
        
        self.test_df = test_df
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration_sec)
        self.music_dir = music_dir
        
        # Torchaudio Log Mel Spectrogram setup
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length if win_length else n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):

        row = self.test_df.iloc[idx]
        file_path = row['filename']
        song_id = file_path.removesuffix('.wav')[-4:]
        
        # Load Audio
        waveform, sr = torchaudio.load(os.path.join(self.music_dir,file_path))
        
        # Convert to Mono (just in case the mashups are stereo)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample to Target Sample Rate
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            
        # Strict 30-Second Trimming/Padding
        current_samples = waveform.shape[1]
        if current_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif current_samples < self.target_samples:
            padding_needed = self.target_samples - current_samples
            waveform = F.pad(waveform, (0, padding_needed))
            
        # Peak Normalization
        abs_max = waveform.abs().max()
        if abs_max > 0:
            waveform = waveform / abs_max
            
        # Log Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amp_to_db(mel_spec)
        
        # Instance Normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-5)
        
        return log_mel_spec, song_id