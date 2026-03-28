import torch
import torchaudio
from torch.utils.data import Dataset
import random
import os

class StemMashupDataset(Dataset):
    def __init__(self, split_data, split_type, music_dir=None, noise_dir=None, samples_per_genre=15000, sample_rate=32000, n_fft=1024, hop_length=512, n_mels=64,f_min=20, f_max=14000, steps_per_epoch= 15000, val_samples=2000, win_length=None):
        
        self.split_type = split_type # 'train' or 'val'
        self.samples_per_genre = samples_per_genre
        self.genres = sorted(list(split_data[split_type].keys()))
        self.num_genres = len(self.genres)
        self.total_samples = self.samples_per_genre * self.num_genres
        self.music_dir = music_dir

        if self.split_type == 'train':
            self.epoch_length = min(steps_per_epoch, self.total_samples)
        else:
            self.epoch_length = val_samples
        
        self.genre_songs = split_data[split_type]
        self.stems = ['vocals.pt', 'drums.pt', 'bass.pt', 'other.pt']
        
        # freq_mask_param=15 means it can mask up to 5 contiguous mel bins
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=5)
        # time_mask_param=35 means it can mask out chunks of time
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)

        # Torchaudio Log Mel Spectrogram setup
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length= win_length if win_length else n_fft, # Defaults to n_fft if not provided
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        
        # Pre-load noise paths (only if training)
        self.noise_paths = []
        if self.split_type == 'train' and noise_dir:
            self.noise_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.pt')]
            
        # Validation set is FIXED every epoch
        self.val_combinations = []
        if self.split_type == 'val':
            # Generate deterministic combinations once during init
            for i in range(self.epoch_length):
                genre_idx = i % self.num_genres
                genre = self.genres[genre_idx]
                
                # Pick 4 random songs from the validation set of this genre
                choices = random.choices(self.genre_songs[genre], k=4)
                self.val_combinations.append((genre, choices))

    def __len__(self):
        # To prevent massive 150k train in one
        return self.epoch_length

    def __getitem__(self, idx):
        # Interleaving genres ensures exactly equal class distribution in the dataset
        genre_idx = idx % self.num_genres
        
        if self.split_type == 'train':
            genre = self.genres[genre_idx]
            # Randomly sample songs for each stem on the fly
            selected_songs = random.choices(self.genre_songs[genre], k=4)
        else:
            # Fetch the pre-computed fixed combination for validation
            genre, selected_songs = self.val_combinations[idx]
            
        # Load and stack stems
        audio_mashup = 0
        for song_path, stem_name in zip(selected_songs, self.stems):
            stem_tensor = torch.load(os.path.join(self.music_dir, song_path, stem_name))
            audio_mashup += stem_tensor # Assuming shape is [channels, time]

        # Environmental Noise Augmentation (Train Only)
        if self.split_type == 'train' and self.noise_paths:
            noise_path = random.choice(self.noise_paths)
            noise_tensor = torch.load(noise_path)
            intensity = random.uniform(0.1, 0.5)
            
            audio_len = audio_mashup.shape[-1]
            noise_len = noise_tensor.shape[-1]
            
            if noise_len <= audio_len:
                # Noise is shorter: add to random position
                start_pos = random.randint(0, audio_len - noise_len)
                audio_mashup[..., start_pos:start_pos+noise_len] += noise_tensor * intensity
            else:
                # Noise is longer: crop noise and add
                start_pos = random.randint(0, noise_len - audio_len)
                audio_mashup += noise_tensor[..., start_pos:start_pos+audio_len] * intensity

        # Normalization (Divide by absmax)
        abs_max = audio_mashup.abs().max()
        if abs_max > 0:
            audio_mashup = audio_mashup / abs_max
            
        # Convert to Log Mel Spectrogram
        mel_spec = self.mel_transform(audio_mashup)
        log_mel_spec = self.amp_to_db(mel_spec)

        # NOrmalisation
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-5)

        if self.split_type == 'train':
            # These things act as Dropout regularisation in training
            log_mel_spec = self.freq_mask(log_mel_spec)
            log_mel_spec = self.time_mask(log_mel_spec)
        
        return log_mel_spec, genre_idx