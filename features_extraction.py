
import warnings
import librosa 
warnings.filterwarnings('ignore')

#  FONCTION D'EXTRACTION
def extraire_features(chemin_fichier):
    import librosa 
    import numpy as np
    try:
        y, sr = librosa.load(chemin_fichier, duration=30)
    except Exception:
        return None
    if len(y) == 0:
        return None
    #  Extraction du Tempo (cas particulier car c'est une valeur unique)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features = {'tempo': float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)} # cas ou on a un array aussi géré
     
    #  Dictionnaire de toutes les variables descriptives à extraire
    features_audio = {
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y),
        'rms': librosa.feature.rms(y=y), # Énergie/Volume global du morceau
        'chroma_stft':librosa.feature.chroma_stft(y=y,sr=sr),
        'spectral_bandwidth':librosa.feature.spectral_bandwidth(y=y,sr=sr),
    }
  
    #  Boucle magique : Calcule automatiquement la moyenne et l'écart-type pour chaque variable
    for nom, descriptors in features_audio.items():
        features[f'{nom}_mean'] = float(np.mean(descriptors))
        features[f'{nom}_std'] = float(np.std(descriptors))
        
    #  Extraction automatique des 13 MFCC (Moyenne + Écart-type pour chacun)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(1, 13):                    # On ignore l'indice 0 (énergie brute)
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i}_std'] = float(np.std(mfccs[i])) # on met a jour features avec les variables 


    #deux signaux: harmonique / percussif 
    y_harm, y_perc = librosa.effects.hpss(y)
    features['perceptr_std'] = float(np.std(y_perc)) 
    features['perceptr_mean'] = float(np.mean(y_perc)) 
    features['harmony_mean'] = float(np.mean(y_harm)) 
    features["harmony_std"] = float(np.std(y_harm)) 

    # Ajout de la taille
    features['length'] = float(librosa.get_duration(y=y, sr=sr))
        
    return features

