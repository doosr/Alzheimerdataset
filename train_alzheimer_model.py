"""
Script Python pour l'entraînement du modèle de classification Alzheimer
Ce script peut être exécuté directement sans Jupyter Notebook

Usage:
    python train_alzheimer_model.py --epochs 50 --batch_size 32
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
np.random.seed(42)
tf.random.set_seed(42)

# Chemins
BASE_DIR = Path('AlzheimerDataset')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR = BASE_DIR / 'test'

MODEL_DIR = Path('AlzheimerModel')
MODELS_DIR = MODEL_DIR / 'models'
GRAPHS_DIR = MODEL_DIR / 'graphs'
LOGS_DIR = MODEL_DIR / 'logs'

# Créer les dossiers
for directory in [MODELS_DIR, GRAPHS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Classes
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = len(CLASSES)


def create_cnn_model(input_shape=(176, 176, 3), num_classes=4):
    """Crée le modèle CNN"""
    model = Sequential([
        # Bloc 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Couches denses
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Sortie
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_training_history(history, save_path):
    """Trace et sauvegarde l'historique d'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val', linewidth=2)
    axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val', linewidth=2)
    axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Trace et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matrice de confusion sauvegardée: {save_path}")


def main(args):
    """Fonction principale"""
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION ALZHEIMER")
    print("="*70)
    
    # Paramètres
    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    
    print(f"\nParamètres:")
    print(f"  - Taille d'image: {IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - GPU disponible: {tf.config.list_physical_devices('GPU')}")
    
    # Générateurs de données
    print("\n" + "="*70)
    print("PRÉPARATION DES DONNÉES")
    print("="*70)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"  - Images d'entraînement: {train_generator.samples}")
    print(f"  - Images de validation: {validation_generator.samples}")
    print(f"  - Images de test: {test_generator.samples}")
    
    # Créer le modèle
    print("\n" + "="*70)
    print("CONSTRUCTION DU MODÈLE")
    print("="*70)
    
    model = create_cnn_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"  - Paramètres totaux: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            filename=str(LOGS_DIR / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    # Entraînement
    print("\n" + "="*70)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("="*70)
    
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    print("\n" + "="*70)
    print("ÉVALUATION SUR LE TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        test_generator, verbose=1
    )
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    print(f"\n  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall:    {test_recall:.4f}")
    print(f"  Test F1-Score:  {test_f1:.4f}")
    
    # Prédictions et matrice de confusion
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    
    print("\n" + "="*70)
    print("RAPPORT DE CLASSIFICATION")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Sauvegardes
    print("\n" + "="*70)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*70)
    
    # Sauvegarder le modèle
    model.save(str(MODELS_DIR / 'alzheimer_model_final.h5'))
    print(f"✓ Modèle sauvegardé: {MODELS_DIR / 'alzheimer_model_final.h5'}")
    
    # Sauvegarder les graphiques
    plot_training_history(history, GRAPHS_DIR / 'training_history.png')
    plot_confusion_matrix(y_true, y_pred, CLASSES, GRAPHS_DIR / 'confusion_matrix.png')
    
    # Sauvegarder l'historique
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(LOGS_DIR / 'training_history.csv', index=False)
    print(f"✓ Historique sauvegardé: {LOGS_DIR / 'training_history.csv'}")
    
    print("\n" + "="*70)
    print("✓✓✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ✓✓✓")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entraînement du modèle Alzheimer')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille du batch')
    parser.add_argument('--img_size', type=int, default=176, help='Taille des images')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
