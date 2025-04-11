import os
import numpy as np
import pandas as pd
import librosa 
import soundfile as sf 
import matplotlib.pyplot as plt 
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

def load_audio_file(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None) #sr = none means load audio with original sample rate
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None
    
def remove_silence(audio_data, sample_rate, threshold = 0.01):
    #audio_data: audio time series
    #sample_rate int: sample rate of the audio file
    #threshold: threshold for silence removal

    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db = 20, frame_length=512, hop_length=64)
    return trimmed_audio

def normalize_volume(audio_data):
    return librosa.util.normalize(audio_data)

def extract_features(audio_data ,sample_rate, n_mfcc=13, n_mels=128, fmax=8000):
    features = {}

    #frame paramteres
    hop_length = int(sample_rate * 0.010) # 10ms
    win_length = int(sample_rate * 0.025) # 25ms

    # 1. time domain features
    #zero crossing rate-  rate at which a signal changes its sign from positive to negative
    #High ZCR → Unvoiced sounds (e.g., consonants like /s/, /f/)
    #Low ZCR → Voiced sounds (e.g., vowels like /a/, /o/)
    #Higher ZCR → Angry, excited speech
    #Lower ZCR → Calm, neutral speech
    #High ZCR → Noisy or percussive music (e.g., rock, hip-hop)
    #Low ZCR → Smooth melodies (e.g., classical, jazz)

    zrc = librosa.feature.zero_crossing_rate(audio_data, hop_length = hop_length)[0]
    features["zrc_mean"] = np.mean(zrc)
    features['zrc_std'] = np.std(zrc)
    features['zrc-max'] = np.max(zrc)

    # 2. RMS energy - useful for stress and emphasis detection
    # Higher for louder sounds, correlates with emotional intensity
    rms  = librosa.feature.rms(y = audio_data, hop_length = hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)

    #mfccs - Mel-frequency cepstral coefficients
    # Captures vocal tract configuration (how we shape our mouth/throat)
    
    mfcc_coeff = librosa.feature.mfcc(
        y = audio_data,
        sr = sample_rate,
        n_mfcc = n_mfcc,
        hop_length = hop_length,
        win_length = win_length,
        n_mels = n_mels,
        fmax = fmax,
    )

    for i in range(n_mfcc):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc_coeff[i])
        features[f'mfcc{i+1}_std'] = np.std(mfcc_coeff[i])
        features[f'mfcc{i+1}_skew'] = skew(mfcc_coeff[i]) #bend of graph, left or right symmertic 
        features[f'mfcc{i+1}_kurt'] = kurtosis(mfcc_coeff[i]) #peakedness of the distribution, how much data is in the tails of the distribution

    #delta = velocity delta-delta = acceleration of mfcc, how feature changes over time
    mfcc_delta = librosa.feature.delta(mfcc_coeff)
    mfcc_delta_delta = librosa.feature.delta(mfcc_coeff, order=2)

    for i in range(n_mfcc):
        features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])
        features[f'mfcc{i+1}_delta2_mean'] = np.mean(mfcc_delta_delta[i])
        features[f'mfcc{i+1}_delta2_std'] = np.std(mfcc_delta_delta[i])


    #spectral centroid - center of mass of the spectrum, indicates brightness of sound
    #Higher centroid → Brighter sound (e.g., cymbals, flutes)
    #Lower centroid → Darker sound (e.g., bass, cello)
    #Higher centroid → Happy, excited speech
    #Lower centroid → Sad, calm speech
    cent = librosa.feature.spectral_centroid(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)


    #spectral bandwidth - width of the spectrum, indicates timbre of sound
    #Higher bandwidth → Wider sound (e.g., brass, distorted guitars)
    #Lower bandwidth → Narrower sound (e.g., strings, flutes)
    #Higher bandwidth → Complex, rich sound
    #Lower bandwidth → Simple, pure sound
    bandwidth = librosa.feature.spectral_bandwidth(y= audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)

    #spectral contrast - difference in amplitude between peaks and valleys in the spectrum
    #Higher contrast → Richer sound (e.g., piano, guitar)
    #Lower contrast → Sparser sound (e.g., flute, clarinet)
    #Higher contrast → Complex, rich sound
    #Lower contrast → Simple, pure sound
    contrast = librosa.feature.spectral_contrast(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)

    #spectral rolloff - frequency below which a certain percentage of the total spectral energy is contained
    #Higher rolloff → More energy in higher frequencies (e.g., distorted guitars, brass)
    #Lower rolloff → More energy in lower frequencies (e.g., bass, cello)
    #Higher rolloff → Complex, rich sound
    #Lower rolloff → Simple, pure sound
    rolloff = librosa.feature.spectral_rolloff(y = audio_data, sr = sample_rate, hop_length = hop_length)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)


    # 3. voice-specific features
    #pitch - perceived frequency of a sound, indicates pitch of voice
    #Fundamental frequency - varies with emotion (higher for happy/surprised)  

    if len(audio_data) > 0:
        try:
            pitches, magnitude = librosa.piptrack(y = audio_data, sr = sample_rate, hop_length = hop_length, fmin= 50, fmax = 1600) #min and max human voice frequency

            pitches_values = []
            for i in range(magnitude.shape[1]):
                index = magnitude[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitches_values.append(pitch)

            if pitches_values:
                features['pitch_mean'] = np.mean(pitches_values)
                features['pitch_std'] = np.std(pitches_values)
                features['pitch_max'] = np.max(pitches_values)
                features['pitch_min'] = np.min(pitches_values)

        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0

    # 4, rhythm features
    #tempo - speed of the music, indicates energy level
    #Higher tempo → Faster rhythm (e.g., dance, electronic music)
    #Lower tempo → Slower rhythm (e.g., ballads, classical music)
    #Higher tempo → Excited, energetic speech
    #Lower tempo → Calm, relaxed speech
    onset_env = librosa.onset.onset_strength(y = audio_data, sr = sample_rate, hop_length = hop_length)
    tempo = librosa.beat.tempo(onset_envelope = onset_env, sr = sample_rate, hop_length = hop_length)[0]
    features['tempo'] = tempo

    # 5. chroma features
    #Chroma features - energy distribution across 12 pitch classes, indicates harmony and tonality
    #Higher chroma features → Richer harmony (e.g., complex chords)
    #Lower chroma features → Simpler harmony (e.g., single notes)

    chroma= librosa.feature.chroma_stft(y = audio_data, sr = sample_rate, hop_length = hop_length)
    for i in range(12):
        features[f"chroma{i+1}_mean"] = np.mean(chroma[i])
        features[f"chroma{i+1}_std"] = np.std(chroma[i])

    return features


# def save_features_to_csv(features, output_path):
#     # Create a DataFrame from the features dictionary
#     df = pd.DataFrame([features])

#     # Save the DataFrame to a CSV file
#     df.to_csv(output_path, index=False)

def process_audio_file(file_path, visualize = False):
    #complete processing pipeline for single audio file:
    #1. load audio file
    #2. remove silence
    #3. normalize volume
    #4. extract features
    #5. visualize(optional)

    #returns: features_dict, metadeta_dict

    #parse file name for metadata
    #ravdess naming: modality - voice channel - emotion - intensity - statement - repetition - actor
    file_name = os.path.basename(file_path)
    file_parts = file_name.split("-")
    
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    metadata = {}

    if len(file_parts) >= 7:
        metadata['file_path'] = file_path
        metadata['modality'] = file_parts[0]
        metadata['voice_channel'] = file_parts[1]
        metadata['emotion'] = emotion_labels.get(file_parts[2], 'unknown')
        metadata['intensity'] = file_parts[3]
        metadata['statement'] = file_parts[4]
        metadata['repetition'] = file_parts[5]
        metadata['actor'] = file_parts[6].split('.')[0]
        metadata['gender'] = 'female' if int(metadata['actor']) % 2 == 0 else 'male'

    else:
        print(f"Invalid file name format: {file_name}")
        return None, None

    audio_data, sample_rate = load_audio_file(file_path)
    if audio_data is None:
        return None, metadata

    #preprocess audio
    # 1. remove silence
    audio_data = remove_silence(audio_data, sample_rate)
    # 2. normalize volume
    audio_data = normalize_volume(audio_data)

    # 3. extract features   
    features = extract_features(audio_data, sample_rate)


    if visualize: 

        #create a directoy for visualizations
        visualization_dir = Path("d:/multimodal emotion recognition system/processed/audio_visualizations") 
        visualization_dir.mkdir(parents=True, exist_ok=True)

        # Create actor+emotion directory
        actor_emotion_dir = visualization_dir / f"actor_{metadata['actor']}_{metadata['emotion']}"
        actor_emotion_dir.mkdir(exist_ok=True)
        
        # Base filename for plots
        base_filename = file_name.replace('.wav', '')

        # Plot and save the waveform
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title(f"Waveform of - {metadata['emotion']}({metadata['intensity']})")

        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, hop_length=512, n_fft=2048)), ref=np.max)
        librosa.display.specshow(D, sr = sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')

        plt.tight_layout()
        plt.savefig(actor_emotion_dir / f"{base_filename}_waveform.png")
        plt.close()

        #plot mfcc
        plt.figure(figsize=(12, 6))
        mfcc_coeff = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

        librosa.display.specshow(mfcc_coeff, sr = sample_rate, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.savefig(actor_emotion_dir / f"{base_filename}_mfcc.png")
        plt.close()

    return features, metadata


def process_audio_directory(directory_path, output_path, max_files = None, visualize_sample = False):
    #Process all audio files in a directory and its subdirectories.
    #directory_path: path to the directory containing audio files
    #output_path: path to save the features and metadata
    #max_files: maximum number of files to process (for testing purposes)
    #visualize_sample: whether to visualize a sample of the audio files

    all_features = []
    all_metadata = []
    File_count = 0

    wav_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    wav_files.sort()

    if max_files is not None:
        wav_files = wav_files[:max_files]

    for file_path in tqdm(wav_files, desc="Processing audio files"):
        vis_this_file = visualize_sample and File_count % 50 == 0
        features, metadata = process_audio_file(file_path, visualize=vis_this_file)

        if features is not None and metadata is not None:
            all_features.append(features)
            all_metadata.append(metadata)
        File_count += 1

    print(f"Successfully processed {len(all_features)} out of {len(wav_files)} files")

    # Create DataFrames for features and metadata
    features_df = pd.DataFrame(all_features)
    metadata_df = pd.DataFrame(all_metadata)

    # Save features and metadata to CSV files
    os.makedirs(output_path, exist_ok=True)
    features_df.to_csv(os.path.join(output_path, "audio_features.csv"), index = False)
    metadata_df.to_csv(os.path.join(output_path, "audio_metadata.csv"), index = False)

    combine_df = pd.concat([features_df, metadata_df], axis=1)
    combine_df.to_csv(os.path.join(output_path, "audio_combined.csv"), index = False)

    print(f"Features and metadata saved to {output_path}")


    return features_df, metadata_df

def analyze_features(features_df, metadata_df, output_path):
    #Analyze extrated features and generate visualizations
    #features_df: DataFrame containing features
    #metadata_df: DataFrame containing metadata
    #output_path: path to save the visualizations

    analysis_dir = Path(output_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    #combine data for analysis
    combined_df = pd.concat([features_df, metadata_df], axis=1)

    # 1. distribution of emotion classes 
    plt.figure(figsize=(12, 6))
    emotion_counts = combined_df['emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Emotions in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(analysis_dir / "emotion_distribution.png")
    plt.close()

    # 2. gender distribution
    plt.figure(figsize=(10, 5))
    gender_counts = combined_df['gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.title('Gender Distribution in Dataset')
    plt.tight_layout()
    plt.savefig(analysis_dir / "gender_distribution.png")
    plt.close()

    # 3. Feature distribution by emotion (for key features)
    key_features = ['mfcc1_mean', 'mfcc2_mean', 'spectral_centroid_mean', 'zrc_mean', 'rms_mean']

    for feature in key_features:
        if feature in features_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='emotion', y=feature, data=combined_df)
            plt.title(f'Distribution of {feature} by Emotion')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{feature}_by_emotion.png")
            plt.close()
    
    # 4. Correlation matix of features
    plt.figure(figsize = (20, 16))
    numeric_features = features_df.select_dtypes(include = [np.number])
    correlation_matrix = numeric_features.corr()
    sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = False, square = True)
    plt.title("Features correlation Matrix")
    plt.tight_layout()
    plt.savefig(analysis_dir / "features_correlation_matrix.png")
    plt.close()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    pca_df['emotion'] = metadata_df['emotion']

    plt.figure(figsize = (12, 8))
    sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'emotion', data = pca_df, palette = 'viridis')
    plt.title("PCA of Audio Features by emotion")
    plt.tight_layout()
    plt.savefig(analysis_dir / "PCA_visualization.png")
    plt.close()

    with open(analysis_dir / 'feature_analysis.txt', 'w') as f:
        f.write("Audio Feature Analysis Summary\n")
        f.write("============================\n\n")
        f.write(f"Total samples: {len(combined_df)}\n")
        f.write(f"Emotion distribution:\n{emotion_counts.to_string()}\n\n")
        f.write(f"Gender distribution:\n{gender_counts.to_string()}\n\n")
        f.write(f"Feature statistics:\n{features_df.describe().to_string()}\n\n")
        f.write(f"PCA explained variance ratio: {pca.explained_variance_ratio_}\n")


if __name__ == "__main__":
    from data_setup import AUDIO_SONG_PATH, AUDIO_SPEECH_PATH, PROCESSED_PATH

    speech_output = PROCESSED_PATH / "audio_features" / "speech"
    song_output = PROCESSED_PATH / "audio_features" / "song"
    
    print("Processing speech audio files...")
    speech_features, speech_metadata = process_audio_directory(AUDIO_SPEECH_PATH, speech_output, visualize_sample=True)


    print("Processing song audio files...")
    song_features, song_metadata = process_audio_directory(AUDIO_SONG_PATH, song_output, visualize_sample=True)

    print("Analyzing speech features...")
    analyze_features(speech_features, speech_metadata, speech_output)
    
    print("Analyzing song features...")
    analyze_features(song_features, song_metadata, song_output)
    
    print("Audio preprocessing completed successfully!")

