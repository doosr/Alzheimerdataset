# 🧠 Classification Alzheimer - Projet Complet

Ce projet contient un notebook Jupyter complet pour la classification des stades de la maladie d'Alzheimer à partir d'images IRM cérébrales.

## 📋 Table des matières

- [Structure du projet](#structure-du-projet)
- [Dataset](#dataset)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture du modèle](#architecture-du-modèle)
- [Résultats](#résultats)
- [Prédictions](#prédictions)

## 📁 Structure du projet

```
fgg/
├── AlzheimerDataset/
│   ├── train/
│   │   ├── MildDemented/
│   │   ├── ModerateDemented/
│   │   ├── NonDemented/
│   │   └── VeryMildDemented/
│   └── test/
│       ├── MildDemented/
│       ├── ModerateDemented/
│       ├── NonDemented/
│       └── VeryMildDemented/
├── AlzheimerModel/
│   ├── models/          # Modèles sauvegardés (.h5, .json)
│   ├── logs/            # Historiques d'entraînement (.csv)
│   └── graphs/          # Visualisations (.png)
├── AlzheimerClassification.ipynb  # Notebook principal
├── requirements.txt     # Dépendances Python
└── README.md           # Ce fichier
```

## 🗂️ Dataset

Le dataset contient des images IRM cérébrales classées en **4 catégories** :

| Classe | Description | Dossier |
|--------|-------------|---------|
| **NonDemented** | Aucun signe de démence | `NonDemented/` |
| **VeryMildDemented** | Démence très légère | `VeryMildDemented/` |
| **MildDemented** | Démence légère | `MildDemented/` |
| **ModerateDemented** | Démence modérée | `ModerateDemented/` |

## 🔧 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer Jupyter Notebook

```bash
jupyter notebook AlzheimerClassification.ipynb
```

## 🚀 Utilisation

### Étapes complètes dans le notebook

Le notebook `AlzheimerClassification.ipynb` contient **19 étapes détaillées** :

1. ✅ **Importation des bibliothèques**
2. ✅ **Configuration des chemins et paramètres**
3. ✅ **Exploration des données**
4. ✅ **Visualisation de la distribution**
5. ✅ **Visualisation d'exemples d'images**
6. ✅ **Préparation des données avec augmentation**
7. ✅ **Construction du modèle CNN**
8. ✅ **Compilation du modèle**
9. ✅ **Configuration des callbacks**
10. ✅ **Entraînement du modèle**
11. ✅ **Visualisation de l'historique**
12. ✅ **Évaluation sur le test set**
13. ✅ **Matrice de confusion**
14. ✅ **Rapport de classification**
15. ✅ **Sauvegarde du modèle**
16. ✅ **Fonction de prédiction**
17. ✅ **Test de prédiction**
18. ✅ **Résumé final**
19. ✅ **Charger un modèle sauvegardé**

### Exécution complète

Exécutez toutes les cellules dans l'ordre pour :
- Explorer et visualiser le dataset
- Entraîner le modèle CNN
- Évaluer les performances
- Sauvegarder le modèle
- Faire des prédictions

## 🏗️ Architecture du modèle

### Modèle CNN personnalisé

```
- 4 blocs convolutionnels (32, 64, 128, 256 filtres)
- Batch Normalization après chaque convolution
- MaxPooling et Dropout entre les blocs
- 2 couches denses (512, 256 neurones)
- Couche de sortie softmax (4 classes)
```

### Paramètres d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Taille d'image | 176x176 |
| Batch size | 32 |
| Epochs | 50 (avec EarlyStopping) |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

### Data Augmentation

- ✅ Rotation aléatoire (±20°)
- ✅ Décalage horizontal/vertical (±20%)
- ✅ Retournement horizontal
- ✅ Zoom aléatoire (±20%)
- ✅ Cisaillement (±20%)

### Callbacks

| Callback | Fonction |
|----------|----------|
| **ModelCheckpoint** | Sauvegarde du meilleur modèle |
| **EarlyStopping** | Arrêt si pas d'amélioration (patience=10) |
| **ReduceLROnPlateau** | Réduction du learning rate (patience=5) |
| **TensorBoard** | Visualisation en temps réel |
| **CSVLogger** | Historique en CSV |

## 📊 Résultats

Les résultats sont sauvegardés dans `AlzheimerModel/models/training_summary.txt`.

### Métriques évaluées

- ✅ **Accuracy** (précision globale)
- ✅ **Precision** (taux de vrais positifs)
- ✅ **Recall** (taux de rappel)
- ✅ **F1-Score** (moyenne harmonique)
- ✅ **Confusion Matrix** (matrice de confusion)
- ✅ **Classification Report** (rapport détaillé par classe)

### Visualisations générées

Toutes les visualisations sont sauvegardées dans `AlzheimerModel/graphs/` :

- 📈 `class_distribution.png` - Distribution des classes
- 🖼️ `sample_images.png` - Exemples d'images par classe
- 📉 `training_history.png` - Courbes d'apprentissage (accuracy, loss, precision, recall)
- 🎯 `confusion_matrix.png` - Matrice de confusion
- 📊 `confusion_matrix_normalized.png` - Matrice de confusion normalisée

## 🔮 Prédictions

### Utiliser le modèle pour de nouvelles images

```python
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('AlzheimerModel/models/alzheimer_model_final.h5')

# Faire une prédiction
result = predict_alzheimer_stage('path/to/mri_image.jpg', model)

print(f"Prédiction: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2f}%")
print(f"Probabilités: {result['probabilities']}")
```

### Visualiser une prédiction

```python
# Affiche l'image avec les probabilités
visualize_prediction('path/to/mri_image.jpg', model)
```

## 📦 Fichiers sauvegardés

### Modèles

| Fichier | Description |
|---------|-------------|
| `alzheimer_model_final.h5` | Modèle complet final |
| `best_model.h5` | Meilleur modèle (val_accuracy max) |
| `alzheimer_weights.h5` | Poids uniquement |
| `model_architecture.json` | Architecture en JSON |

### Logs

| Fichier | Description |
|---------|-------------|
| `training_log.csv` | Historique complet par epoch |
| `training_history.csv` | Métriques d'entraînement |
| `training_summary.txt` | Résumé textuel |

## 🎯 Prochaines étapes

### Améliorations possibles

1. **Transfer Learning** 🔄
   - Essayer VGG16, ResNet50, InceptionV3
   - Fine-tuning des couches pré-entraînées

2. **Optimisation** ⚡
   - Hyperparameter tuning (GridSearch, RandomSearch)
   - Essayer différentes architectures
   - Tester d'autres optimizers (SGD, RMSprop)

3. **Déploiement** 🚀
   - Créer une API Flask/FastAPI
   - Développer une application web (Streamlit, Gradio)
   - Application mobile (TensorFlow Lite)

4. **Analyse** 🔍
   - Étudier les erreurs de classification
   - Grad-CAM pour visualiser les zones importantes
   - Analyse des faux positifs/négatifs

## 📝 Notes

- Le modèle utilise la **normalisation** (division par 255) pour les images
- Les images sont **redimensionnées à 176x176** pixels
- Format supporté : **JPG, PNG**
- GPU recommandé pour l'entraînement (mais fonctionne sur CPU)

## 🤝 Support

Pour toute question ou problème :
1. Vérifier que toutes les dépendances sont installées
2. S'assurer que le dataset est dans le bon format
3. Vérifier les chemins dans le notebook

## 📄 Licence

Ce projet est fourni à des fins éducatives.

---

**Créé avec ❤️ pour la recherche sur Alzheimer**
"# Alzheimerdataset" 
"# Alzheimerdataset" 
