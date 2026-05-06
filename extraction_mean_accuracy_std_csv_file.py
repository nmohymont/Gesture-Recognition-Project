import pandas as pd
import numpy as np
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_DOMAIN4_D = os.path.join(BASE_DIR, "Resultats_statistiques", "resultats_user_indep_domain4_downsampled_PyTorch.csv")
DATA_DIR_DOMAIN1_D = os.path.join(BASE_DIR, "Resultats_statistiques", "resultats_user_indep_domain1_downsampled_PyTorch.csv")

DATA_DIR_DOMAIN1 = os.path.join(BASE_DIR, "Resultats_statistiques", "resultats_user_indep_domain1_PyTorch.csv")
DATA_DIR_DOMAIN4 = os.path.join(BASE_DIR, "Resultats_statistiques", "resultats_user_indep_domain4_PyTorch.csv")

try:
    # Lecture du fichier CSV
    df = pd.read_csv(DATA_DIR_DOMAIN4_D)
    
    # Vérification que la colonne 'is_correct' existe bien
    if 'is_correct' in df.columns:
        
        # 1. Calcul de l'accuracy moyenne globale et de l'écart-type
        accuracy_globale = df['is_correct'].mean()
        ecart_type_global = df['is_correct'].std()
        
        print("--- Résultats Globaux ---")
        print(f"Accuracy moyenne : {accuracy_globale:.4f} ({accuracy_globale*100:.2f}%)")
        print(f"Écart-type       : {ecart_type_global:.4f}\n")
        
        # 2. Vérification et calcul de l'accuracy par 'subject_id'
        if 'subject_id' in df.columns:
            print("--- Accuracy par Sujet ---")
            
            # On groupe les données par 'subject_id' et on calcule la moyenne de 'is_correct'
            accuracy_par_sujet = df.groupby('subject_id')['is_correct'].mean()
            
            # On parcourt les résultats pour les afficher proprement
            for subject, acc in accuracy_par_sujet.items():
                print(f"Sujet {subject} : {acc:.4f} ({acc*100:.2f}%)")
                
        else:
            print("Note : La colonne 'subject_id' n'existe pas dans ce fichier CSV.")
            
    else:
        print("Erreur : La colonne 'is_correct' n'existe pas dans ce fichier CSV.")

except FileNotFoundError:
    # Correction ici : utilisation de la bonne variable pour le message d'erreur
    print(f"Erreur : Le fichier '{DATA_DIR_DOMAIN4_D}' est introuvable.")