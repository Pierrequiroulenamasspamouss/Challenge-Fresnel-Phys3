import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION DU STYLE GRAPHIQUE (IDENTIQUE AU CODE PIXEL) ---
plt.rcParams['grid.alpha'] = 1.0  # Opaque
plt.rcParams['grid.linewidth'] = 1.0 
plt.rcParams['grid.color'] = 'darkgray'
plt.rcParams['axes.grid'] = True 
plt.style.use('seaborn-v0_8-whitegrid')

###########################################
# 1) CHARGEMENT DES DONNÉES DE RÉFÉRENCE (POUR INCERTITUDES)
###########################################

df_real = pd.read_csv("mesures_reelles.csv")
df_real.columns = df_real.columns.str.strip()

# On récupère la longueur minimale pour aligner tous les tableaux
min_length = len(df_real)
print(f"Alignement sur la longueur minimale de 'mesures_reelles.csv' : {min_length}")

# Récupération de l'incertitude en mm (divisée par 2 comme dans ton code pixel)
uncertainty_mm_array = df_real["Incertitude_mm"].values / 2
# Troncature de l'incertitude
uncertainty_mm_array = uncertainty_mm_array[:min_length]

###########################################
# 2) TRAITEMENT DES DONNÉES IMAGEJ (EN MM)
###########################################

df_img = pd.read_csv("Results.csv")

# Extraction des p et q (alternance des lignes)
# Hypothèse : Results.csv contient déjà les valeurs en mm 
# (car dans le code 1, tu multipliais ces valeurs par le PPU pour avoir des pixels)
p_values = df_img["Length"].iloc[0::2].reset_index(drop=True)
q_values = df_img["Length"].iloc[1::2].reset_index(drop=True)

# Troncature pour correspondre à df_real
p_values = p_values[:min_length]
q_values = q_values[:min_length]

# Création du DataFrame propre
df_img2 = pd.DataFrame({
    "p_mm": pd.to_numeric(p_values, errors='coerce'),
    "q_mm": pd.to_numeric(q_values, errors='coerce'),
    "err_mm": uncertainty_mm_array
})

# Calculs des inverses (1/p et 1/q)
df_img2["inv_p"] = 1 / df_img2["p_mm"]
df_img2["inv_q"] = 1 / df_img2["q_mm"]

# --- CALCUL DES BARRES D'ERREUR (PROPAGATION) ---
# Formule : delta(1/x) = delta(x) / x^2
df_img2["err_inv_q"] = df_img2["err_mm"] / (df_img2["q_mm"] ** 2) /2
df_img2["err_inv_p"] = df_img2["err_mm"] / (df_img2["p_mm"] ** 2) /2

# Régression linéaire : Y = A * X + B
a_img, b_img = np.polyfit(df_img2["inv_q"], df_img2["inv_p"], 1)
focale_img = 1 / b_img

###########################################
# 3) GRAPHIQUE ESTHÉTIQUE (FORMAT PIXEL MAIS EN MM)
###########################################

plt.figure(figsize=(10, 7))

# A. Tracé des points avec barres d'erreur
plt.errorbar(
    df_img2["inv_q"],
    df_img2["inv_p"],
    xerr=df_img2["err_inv_q"],
    yerr=df_img2["err_inv_p"],
    fmt='o',
    markersize=6,            
    markerfacecolor='blue', 
    markeredgecolor='black',
    capsize=4,
    ecolor='gray',
    alpha=0.7,
    label="Mesures ImageJ (mm) ± Incertitude"
)

# B. Tracé de la régression
x_line2 = np.linspace(df_img2["inv_q"].min(), df_img2["inv_q"].max(), 100)
y_line2 = a_img * x_line2 + b_img

plt.plot(x_line2, y_line2, 
         label="Régression linéaire", 
         linewidth=2.5, 
         color='red', 
         linestyle='-')

# C. Étiquettes et Titres (LaTeX)
plt.xlabel(r"$1/q \quad (\mathrm{mm}^{-1})$", fontsize=18)
plt.ylabel(r"$1/p \quad (\mathrm{mm}^{-1})$", fontsize=18)
# plt.title("Loi des lentilles minces – Données ImageJ (mm)", fontsize=16, fontweight='bold')

plt.legend(fontsize=12, frameon=True, shadow=True)

# La grille est gérée par les rcParams définis au début
plt.tight_layout()

# Sauvegarde
output_filename = "Graphiques/ImageJ-mm-incertitudes.png"
# Création du dossier si inexistant
import os
if not os.path.exists('Graphiques'):
    os.makedirs('Graphiques')
    
plt.savefig(output_filename, dpi=600)
plt.show()

###########################################
# 4) AFFICHAGE DES RÉSULTATS
###########################################
print("\n--- RÉSULTATS IMAGEJ (mm) ---")
print(f"Pente (A) = {a_img:.6f}")
print(f"Ordonnée à l'origine (B) = {b_img:.6f}")
print(f"Focale estimée (ImageJ) : f = {focale_img:.2f} mm")
print(f"Graphique sauvegardé sous : {output_filename}")