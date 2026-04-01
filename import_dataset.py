import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "Domain1_csv")

#print(len(os.listdir(DATA_DIR))) validation que tous les chiffiers ont été chargés

def load_dataset(data_dir):
    all_data = []

    #print(os.listdir(DATA_DIR)) print tout les fichier du dossier Dataset/Domain1_csv 
    #pour vérifier que les fichiers sont bien là

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue
        
        name = file.replace(".csv", "").replace("Subject", "")
        parts = name.split("-")

        if len(parts) != 3:
            print(f"Filename {file} does not match expected format. Skipping.")
            continue
    
        subject_id = int(parts[0])
        digit= int(parts[1])
        repetition = int(parts[2])

        filepath = os.path.join(DATA_DIR, file)
        df = pd.read_csv(filepath, skiprows=[0], names=['x','y','z','t'])
    
        df['subject_id'] = subject_id
        df['digit'] = digit
        df['repetition'] = repetition

        all_data.append(df)
    return all_data

dataset = pd.concat(load_dataset(DATA_DIR), ignore_index=True)

print(f"{len(dataset)} fichiers chargés .")
print(f"Dimension du dataset : {dataset.shape}")

print(dataset.head())

# ligne a décommenter pour sauvegarder le dataset traité dans un fichier csv
dataset.to_csv(os.path.join(BASE_DIR, "Dataset", "Aggregated_csv", "Domain1_processed_dataset.csv"), index=False)

#-------------------------------------------
#solution pour charger les data du csv en évitant l'import d'une ligne 0 <x> <y> <z> <t> skiprows=[0]

#df = pd.read_csv(os.path.join(DATA_DIR, "Subject1-0-1.csv"), skiprows=[0], names=['x','y','z','t'])
#print(df.head())
