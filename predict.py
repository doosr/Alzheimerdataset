"""
Script de prédiction pour le modèle de classification Alzheimer
Permet de faire des prédictions sur de nouvelles images IRM

Usage:
    python predict.py --image path/to/image.jpg --model AlzheimerModel/models/alzheimer_model_final.h5
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Classes
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


def predict_alzheimer_stage(image_path, model, img_size=(176, 176)):
    """
    Prédit le stade d'Alzheimer pour une image IRM
    
    Args:
        image_path: Chemin vers l'image
        model: Modèle chargé
        img_size: Taille de l'image
    
    Returns:
        dict: Résultats de la prédiction
    """
    # Charger et prétraiter l'image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASSES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Probabilités pour toutes les classes
    probabilities = {CLASSES[i]: float(predictions[0][i] * 100) for i in range(len(CLASSES))}
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'image': img
    }


def visualize_prediction(image_path, model, save_path=None):
    """
    Visualise la prédiction avec l'image et les probabilités
    """
    result = predict_alzheimer_stage(image_path, model)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Afficher l'image
    axes[0].imshow(result['image'])
    axes[0].set_title(
        f"Prédiction: {result['predicted_class']}\nConfiance: {result['confidence']:.2f}%",
        fontsize=12, 
        fontweight='bold'
    )
    axes[0].axis('off')
    
    # Afficher les probabilités
    classes = list(result['probabilities'].keys())
    probs = list(result['probabilities'].values())
    colors = ['green' if c == result['predicted_class'] else 'gray' for c in classes]
    
    axes[1].barh(classes, probs, color=colors, alpha=0.7)
    axes[1].set_xlabel('Probabilité (%)', fontsize=11)
    axes[1].set_title('Probabilités par classe', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 100)
    
    # Ajouter les valeurs
    for i, (c, p) in enumerate(zip(classes, probs)):
        axes[1].text(p + 1, i, f'{p:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualisation sauvegardée: {save_path}")
    else:
        plt.show()
    
    return result


def main(args):
    """Fonction principale"""
    print("="*70)
    print("PRÉDICTION ALZHEIMER - ANALYSE D'IMAGE IRM")
    print("="*70)
    
    # Vérifier les fichiers
    model_path = Path(args.model)
    image_path = Path(args.image)
    
    if not model_path.exists():
        print(f"❌ ERREUR: Le modèle n'existe pas: {model_path}")
        return
    
    if not image_path.exists():
        print(f"❌ ERREUR: L'image n'existe pas: {image_path}")
        return
    
    print(f"\nChargement du modèle: {model_path}")
    model = load_model(str(model_path))
    print("✓ Modèle chargé avec succès!")
    
    print(f"\nAnalyse de l'image: {image_path}")
    
    # Prédiction
    if args.visualize:
        save_path = args.output if args.output else None
        result = visualize_prediction(str(image_path), model, save_path)
    else:
        result = predict_alzheimer_stage(str(image_path), model)
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("RÉSULTATS DE LA PRÉDICTION")
    print("="*70)
    print(f"\n🧠 Classe prédite: {result['predicted_class']}")
    print(f"📊 Confiance: {result['confidence']:.2f}%")
    print(f"\n📈 Probabilités détaillées:")
    for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 2)
        print(f"  {class_name:20s}: {prob:6.2f}% {bar}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prédiction Alzheimer sur image IRM')
    parser.add_argument('--image', type=str, required=True, help='Chemin vers l\'image à analyser')
    parser.add_argument('--model', type=str, 
                       default='AlzheimerModel/models/alzheimer_model_final.h5',
                       help='Chemin vers le modèle (.h5)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Afficher la visualisation graphique')
    parser.add_argument('--output', type=str, 
                       help='Sauvegarder la visualisation (si --visualize est activé)')
    
    args = parser.parse_args()
    main(args)
