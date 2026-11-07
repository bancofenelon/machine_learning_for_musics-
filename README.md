
# **machine_learning_for_musics**: Analyse Audio & Prédiction de Genres Musicaux

Ce projet utilise le dataset GTZAN (1000 extraits audio de 30 s) pour extraire des descripteurs avec Librosa,analyser ces descripteurs  et entraîner plusieurs modèles de Machine Learning capables de prédire le genre musical d'un morceau.

## Objectifs
* **Analyse exploratoire (EDA)** : extraction et visualisation des descripteurs audio (MFCC, chroma, centroïde, rolloff, contraste spectral, etc.).
* **Classification** : entraînement et comparaison de modèles (SVM, Random Forest, XGBoost...) pour prédire le genre parmi les 10 catégories de GTZAN.
* **Prédiction** : donner un morceau au modèle et obtenir son genre.

## Dataset
1000 pistes de 30 secondes, 10 genres (100 pistes chacun) : blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

## Objectifs 
* **Analyse exploratoire (EDA)** : Extraction et visualisation des caractéristiques audio ( MFCC, Chroma,perceptr,rollof ect.. ).
* **Classification & Prédiction** : Entraînement des modèles (ex: SVM, Random Forest, Xgboost ...) pour prédire automatiquement le genre musical parmi les 10 catégories de GTZAN tout en explorant de près.    

## Dataset
Le dataset GTZAN contient 1000 pistes audio de 30 secondes, réparties en 10 genres (100 pistes par genre) :
* Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

## Installation

```bash
git clone https://github.com/bancofenelon/machine_learning_for_musics-.git
cd machine_learning_for_musics-
python -m venv .venv
.venv\Scripts\activate        #  si windows
pip install -r requirements.txt
```

Télécharge GTZAN depuis [Kaggle](https://www.kaggle.com/datasets/achgls/gtzan-music-genre) et place les fichiers dans `audio_data/`.

## Utilisation

```bash
python features_extraction.py  # préparation de notre module extracteur
jupyter notebook Audio_features_extraction_file.ipynb  # génère my_gtzan_features.csv
jupyter notebook GTZAN.ipynb    # analyse et modèles + prédiction
```


## Structure du projet
* `features_extraction.py` :module simple python crée  contenant une fonction d'extraction des descripteurs (Librosa).
* `Audio_features_extraction_file.ipynb` : applique l'extraction aux 1000 morceaux et produit `my_gtzan_features.csv`.
* `GTZAN.ipynb` : analyse des descripteurs, visualisations, modèles, optimisation des hyperparamètres, feature engineering et prédiction.
* `audio_data/` : dossier audio GTZAN retrouvable sur kaggle `https://www.kaggle.com/datasets/achgls/gtzan-music-genre`.
* `requirements.txt` : dépendances Python.

## Prochaines étapes (En cours)
* Optimisation  avancée des hyperparamètres des modèles.
* Utlisation de données primaires et suffisantes et des résultats pour créer un système de recommandation complet et aussi de **répertoire de musique intéligent**.

 **Auteur** : [bancofenelon](https://github.com)

