import numpy as np
import pandas as pd
import os

def load_domain4_dataset(folder_path):
    """
    Charge le Domain 4 même s'il est au format .txt.

    Les fichiers Domain 4 ont ce format :
    Domain id = 4
    Class id = ...
    User id = ...

    <x>,<y>,<z>,<t>
    ...
    """

    all_data = []
    repetition_counter = {}

    files = sorted(os.listdir(folder_path))

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        domain_id = int(lines[0].split("=")[1].strip())
        class_id = int(lines[1].split("=")[1].strip())
        user_id = int(lines[2].split("=")[1].strip())

        key = (user_id, class_id)

        if key not in repetition_counter:
            repetition_counter[key] = 1
        else:
            repetition_counter[key] += 1

        repetition = repetition_counter[key]

        df = pd.read_csv(file_path, skiprows=4)

        df.columns = (
            df.columns
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
        )

        df["domain_id"] = domain_id
        df["subject_id"] = user_id
        df["digit"] = class_id
        df["repetition"] = repetition
        df["filename"] = filename

        df = df[["subject_id", "digit", "repetition", "x", "y", "z", "t", "domain_id", "filename"]]

        all_data.append(df)

    return all_data

def standardize_group(group):
    coords = group[['x', 'y', 'z']].values 

    gravity_center = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    std[std == 0] = 1 # Sécurité pour éviter la division par zéro
    
    coords_std = (coords - gravity_center) / std
    
    # On ajoute directement les colonnes au groupe
    group['x_std'] = coords_std[:, 0]
    group['y_std'] = coords_std[:, 1]
    group['z_std'] = coords_std[:, 2]

    return group

def count_gestures(df):
    return len(df.groupby(['subject_id', 'digit', 'repetition']))

