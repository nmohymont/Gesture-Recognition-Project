import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_AGGREGATED = os.path.join(BASE_DIR, "Dataset", "Aggregated_csv")

if not os.path.exists(DATA_DIR_AGGREGATED):
    os.makedirs(DATA_DIR_AGGREGATED)

df = pd.read_csv(os.path.join(DATA_DIR_AGGREGATED, "Domain1_processed_dataset.csv"))

def downsample_data(group,factor=2):

    # Supprime les lignes où t n'est pas numérique
    group = group[pd.to_numeric(group['t'], errors='coerce').notna()]

    group['t'] = group['t'].astype(float)

    group = group.sort_values('t').reset_index(drop=True)
    
    #print(group.head())

    N= len(group)

    if N<66: #datapoints min = 32 donc on downsample à partir de 66 points (32*2) pour éviter de perdre trop d'information
        return group

    t_first = int(group['t'].iloc[0])
    t_last = int(group['t'].iloc[-1])
    
    duration = t_last - t_first

    if duration == 0:
        return group
    
    f_original = N / duration
    f_new = f_original / factor

    dt_new = 1.0 / f_new

    target_times = np.arange(t_first, t_last + dt_new * 0.5, dt_new)
    t_values = group['t'].values
    chosen_idx= set()

    for t_target in target_times:
        idx = np.argmin(np.abs(t_values - t_target)) # trouve l'instant théorique idéel le plus proche de ligne réeel
        chosen_idx.add(idx)

    result = group.iloc[sorted(chosen_idx)].reset_index(drop=True)

    if not (32<= len(result) <= 120):
        return group
    
    return result 

print("Sous-echantillonnage des données...")

df_downsampled = (
    df.groupby(['subject_id', 'digit', 'repetition']) 
    .apply(downsample_data)
    .reset_index(drop=True)
)

original_count = len(df) 
downsampled_count = len(df_downsampled)
reduction = (1-downsampled_count/original_count)*100

print(f"Terminé !")
print(f"Points originaux     : {original_count:,}")
print(f"Points sous-échant.  : {downsampled_count:,}")
print(f"Réduction effective  : {reduction:.1f}%")

# 2. Calcul des gestes (combinaisons uniques sujet/digit/rep)
def count_gestures(df):
    return len(df.groupby(['subject_id', 'digit', 'repetition']))

old_gestures = count_gestures(df)
new_gestures = count_gestures(df_downsampled)

print(f"=== BILAN DU PRÉ-TRAITEMENT ===")
print(f"Points (coordonnées) : {original_count:,} -> {downsampled_count:,} ({reduction:.1f}% de réduction)")
print(f"Gestes (séquences)   : {old_gestures} -> {new_gestures} (Conservation: {(new_gestures/old_gestures)*100:.1f}%)")

if old_gestures == new_gestures:
    print("✓ Succès : Tous les gestes ont été conservés.")
else:
    print(f"⚠ Attention : {old_gestures - new_gestures} geste(s) perdu(s).")


# ligne a décommenter pour sauvegarder le dataset traité dans un fichier csv
#df_downsampled.to_csv(os.path.join(BASE_DIR, "Dataset", "Domain1_aggregated_csv", "Domain1_downsampled_dataset.csv"), index=False)

# prendre un subset 
subset = df[(df['subject_id'] == 1) & (df['digit'] == 0)]

for (subject_id, digit,rep), group in subset.groupby(['subject_id', 'digit', 'repetition']):
   print(f"Subject ID: {subject_id}, Digit: {digit}, Number of rows: {len(group)}")


# code pour afficher les points avant et après le sous-échantillonnage pour un groupe spécifique (par exemple, subject_id=1, digit=0)

Subject = 1 
Line_color = 'black'

Digits = sorted(df['digit'].unique())
N_rows = len(Digits)
N_columns = 2



def plot_xy_grid(df, df_downsampled, subject=1, line_color='black'):

    Digits = sorted(df['digit'].unique())
    N_rows = len(Digits)
    N_columns = 2

    fig, axes = plt.subplots(N_rows, N_columns, figsize=(18, 7))
    
    for row_idx, digit in enumerate(Digits):

         # Sélection du groupe original
        df_group = df[
            (df['subject_id'] == subject) &
            (df['digit'] == digit)
        ]

        rep_counts = df_group.groupby("repetition").size()
        best_rep = rep_counts.idxmin() # répétition avec le moins de points (plus représentative du sous-échantillonnage)

        # données originales pour la répétition choisie
        df_orig= df_group[df_group['repetition'] == best_rep].sort_values('t')
        n_orig= len(df_orig)

        # données sous-échantillonnées pour le même groupe
        df_downsampled = df_downsampled[
            (df_downsampled['subject_id'] == subject) &
            (df_downsampled['digit'] == digit) &
            (df_downsampled['repetition'] == best_rep)
        ].sort_values('t')
        n_downsampled = len(df_downsampled)


        for col in ["x", "y"]:
            df_orig[col] = df_orig[col].astype(float)
            df_downsampled[col] = df_downsampled[col].astype(float)

        # Limite d'axes (référence commune pour les deux subplots)
        x_min = min(df_orig["x"].min(), df_downsampled["x"].min())
        x_max = max(df_orig["x"].max(), df_downsampled["x"].max())
        y_min = min(df_orig["y"].min(), df_downsampled["y"].min())
        y_max = max(df_orig["y"].max(), df_downsampled["y"].max())
        margin = 0.05

        # --- Subplot gauche : original ---
        ax_left  = axes[row_idx, 0]
        ax_left.plot(df_orig["x"].values, df_orig["y"].values,
                 color=line_color, linewidth=0.9, alpha=0.85)
        ax_left.scatter(df_orig["x"].values, df_orig["y"].values,
                    color=line_color, s=6, zorder=3)
        ax_left.set_title(
            f"Digit {digit} — Original   [n = {n_orig}]",
            fontsize=9, fontweight="bold", loc="left"
        )
        ax_left.set_xlabel("x", fontsize=7)
        ax_left.set_ylabel("y", fontsize=7)
        ax_left.tick_params(labelsize=6)
        ax_left.grid(True, linewidth=0.3, alpha=0.4)

        ax_left.set_xlim(x_min - margin, x_max + margin)
        ax_left.set_ylim(y_min - margin, y_max + margin)

        # --- Subplot droite : sous-échantillonné ---
        ax_right = axes[row_idx, 1]
        ax_right.plot(df_downsampled["x"].values, df_downsampled["y"].values,
                  color=line_color, linewidth=0.9, alpha=0.85)
        ax_right.scatter(df_downsampled["x"].values, df_downsampled["y"].values,
                     color=line_color, s=6, zorder=3)

        
        ax_right.set_title(
            f"Digit {digit} —    [n = {n_downsampled}]",
            fontsize=9, fontweight="bold", loc="left"
        )
        ax_right.set_xlabel("x", fontsize=7)
        ax_right.set_ylabel("y", fontsize=7)
        ax_right.tick_params(labelsize=6)
        ax_right.grid(True, linewidth=0.3, alpha=0.4)

        ax_right.set_xlim(x_min - margin, x_max + margin)
        ax_right.set_ylim(y_min - margin, y_max + margin)

        
    # --- En-têtes colonnes ---
    orig_title= axes[0, 0].get_title()
    down_title = axes[0, 1].get_title()

    axes[0, 0].set_title(
        f"ORIGINAL  ·  Subject {subject}\n" + axes[0, 0].get_title(),
        fontsize=9, fontweight="bold", loc="left"
    )
    axes[0, 1].set_title(
        f"DOWNSAMPLED  ·  Subject {subject}\n" + axes[0, 1].get_title(),
        fontsize=9, fontweight="bold", loc="left"
    )

    plt.suptitle(
        f"Comparaison original vs sous-échantillonné — Subject {subject}",
        fontsize=13, fontweight="bold", y=1.002
    )
    plt.subplots_adjust(hspace=0.85)   # espace vertical entre lignes (0.0 à 1.0)
    plt.subplots_adjust(wspace=0.15)   # espace horizontal entre colonnes

    plt.show()
    print("Affichage terminé.")


#plot_xy_grid(subject=10)

# plot 3D of 1 subject and 1 digit before and after downsampling the repetition with the least points (most representative of the downsampling)

def plot_3d(df, df_downsampled, subject=1, digit=0, line_color='black'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sélection du groupe original
    df_group = df[
        (df['subject_id'] == subject) &
        (df['digit'] == digit)
    ]

    rep_counts = df_group.groupby("repetition").size()
    best_rep = rep_counts.idxmin() # répétition avec le moins de points (plus représentative du sous-échantillonnage)

    # données originales pour la répétition choisie
    df_orig= df_group[df_group['repetition'] == best_rep].sort_values('t')

    # données sous-échantillonnées pour le même groupe
    df_downsampled = df_downsampled[
        (df_downsampled['subject_id'] == subject) &
        (df_downsampled['digit'] == digit) &
        (df_downsampled['repetition'] == best_rep)
    ].sort_values('t')

    for col in ["x", "y", "z"]:
        df_orig[col] = df_orig[col].astype(float)
        df_downsampled[col] = df_downsampled[col].astype(float)

    ax.plot(df_orig["x"].values, df_orig["y"].values, df_orig["t"].values,
             color=line_color, linewidth=0.9, alpha=0.85, label="Original")
    ax.scatter(df_orig["x"].values, df_orig["y"].values, df_orig["t"].values,
                color=line_color, s=6, zorder=3)

    ax.plot(df_downsampled["x"].values, df_downsampled["y"].values, df_downsampled["t"].values,
              color='red', linewidth=0.9, alpha=0.85, label="Downsampled")
    ax.scatter(df_downsampled["x"].values, df_downsampled["y"].values, df_downsampled["t"].values,
                 color='red', s=6, zorder=3)

    ax.set_title(
        f"3D Gesture - Subject {subject} — Digit {digit} — Rep {best_rep}\n Comparaison n_original {df_orig.shape[0]} vs n_downsampled {df_downsampled.shape[0]}",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("t", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend()
    plt.show()
    print("Affichage 3D terminé.")

#plot_3d(subject=10, digit=0)

# ------------------------------
#Domain4 
# ----------------------------

def txt_to_df(input_path):
    # 1. Extraction de la répétition (dernier chiffre du nom de fichier)
    repetition = int(os.path.basename(input_path).split('.')[0][-1])

    # 2. Lecture des métadonnées (IDs) dans l'en-tête 
    with open(input_path, 'r') as f:
        head = [next(f) for _ in range(6)]
    
    # "Class id = X" est à la ligne 2 (index 1) 
    class_id = int(head[1].split('=')[-1].strip()) 
    # "User id = X" est à la ligne 3 (index 2) 
    user_id = int(head[2].split('=')[-1].strip()) 

    # 3. Chargement des données (x, y, z, t)
    # On saute les 5 premières lignes et on ignore l'accolade finale 
    df = pd.read_csv(input_path, skiprows=5, skipfooter=1, engine='python')
    
    # Nettoyage des colonnes : <x>,<y>,<z>,<t> -> x, y, z, t 
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]

    # 4. AJOUT DES COLONNES DE RÉFÉRENCE
    # On propage les IDs sur toutes les lignes de ce geste
    df['subject_id'] = user_id
    df['class_id'] = class_id
    df['repetition'] = repetition

    return df


#DATA_DIR_DOMAIN4_TEST = os.path.join(BASE_DIR, "Dataset", "Domain4_csv","3001.txt")
#df = txt_to_df(DATA_DIR_DOMAIN4_TEST)

#print(df.head())

DATA_DIR_DOMAIN4 = os.path.join(BASE_DIR, "Dataset", "Domain4_csv")

# Liste pour stocker tous les DataFrames
liste_dfs = []


for file in os.listdir(DATA_DIR_DOMAIN4):
    if file.endswith(".txt") and "(1)" not in file:
        path = os.path.join(DATA_DIR_DOMAIN4, file)
        try:
            # On transforme et on ajoute à la liste
            df_geste = txt_to_df(path)
            liste_dfs.append(df_geste)
        except Exception as e:
            print(f"Erreur sur {file}: {e}")

# FUSION FINALE : Un seul DataFrame pour les 1000 fichiers
if liste_dfs:
    df_final = pd.concat(liste_dfs, ignore_index=True)
    print(f"Extraction terminée ! Taille totale : {df_final.shape}")
else:
    print("Aucun fichier trouvé.")

        