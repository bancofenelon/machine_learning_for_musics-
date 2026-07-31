
# **machine_learning_for_musics**: Analyse Audio & Prédiction de Genres Musicaux

Ce projet utilise le célèbre dataset GTZAN(features_csv_30_sec)  pour analyser des fichiers audio  et entraîner plusieurs modèles de Machine Learning  capables de prédire le genre musical d'un morceau .

## Objectifs 
* **Analyse exploratoire (EDA)** : Extraction et visualisation des caractéristiques audio ( MFCC, Chroma,perceptr,rollof ect.. ).
* **Classification & Prédiction** : Entraînement des modèles (ex: SVM, Random Forest, Xgboost ...) pour prédire automatiquement le genre musical parmi les 10 catégories de GTZAN tout en explorant de près.    

## Dataset
Le dataset GTZAN contient 1000 pistes audio de 30 secondes, réparties en 10 genres (100 pistes par genre) :
* Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.


## Installation et Dépendances

Le projet est entièrement documenté et exécutable dans un notebook Python.

```bash
# 1. Cloner le projet
git clone git@github.com:bancofenelon/machine_learning_for_musics.git
cd machine_learning_for_musics

# 2. Installer les bibliothèques requises
pip install librosa numpy pandas matplotlib seaborn scikit-learn notebook tqdm xgboost librosa
```

## Utilisation

Pour explorer l'analyse et lancer les prédictions, ouvrez le notebook principal :

```bash
jupyter notebook GTZAN.ipynb

```

## Structure du Projet
* `feature_extraction.py` : module simpliste python crée  contenant une fonction responsable de l'extraction de features ( avec Librosa)
* `GTZAN.ipynb` : Notebook contenant l'analyse des descripteurs audios, des visualisations et les modèles de prédiction appuyé par des analyses de leurs performances,d'optimisations
des hyperparamètres pour les modèles et de feature engenering.
* `Data` : Dossier contenant le dataset GTZAN (fichiers audio et fichiers CSV de features :`features_csv_30_sec`).
* 'requirements.txt' : Liste des dépendances Python nécessaires.

## Prochaines étapes (En cours)
* Optimisation  avancée des hyperparamètres des modèles.
* Déploiement d'une interface simple (Streamlit ou Gradio) pour tester le modèle avec vos propres fichiers audio.

 **Auteur** : [bancofenelon](https://github.com)

