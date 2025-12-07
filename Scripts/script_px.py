import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. GÉNÉRATION DE results_pixels.csv (avec PPU séquentiel) ---

# Load and Clean scales.csv to get sequential PPU and PPU_avg
scales_df = pd.read_csv('scales.csv', delimiter=';')
scales_df.columns = scales_df.columns.str.strip()
ppu_col_name = 'PPU (pixel-per-unit)'
scales_df[ppu_col_name] = (
    scales_df[ppu_col_name]
    .astype(str)
    .str.replace(',', '.', regex=False)
    .astype(float)
)
ppu_values_full = scales_df[ppu_col_name].tolist()
ppu_avg = scales_df[ppu_col_name].mean()
mm_per_pixel_factor = 1 / ppu_avg

# Determine PPU sequence
results_df = pd.read_csv('Results.csv')
results_df.columns = results_df.columns.str.strip()
num_results = len(results_df) # 58
num_scales_needed = num_results // 2 # 29
ppu_values = ppu_values_full[:num_scales_needed] # 29 sequential PPU values

# Generate sequential PPU arrays for conversion
sequential_ppu = np.repeat(ppu_values, 2)
sequential_ppu_squared = sequential_ppu ** 2

# Perform Conversion to create results_pixels.csv
results_pixels_df = results_df.copy()
area_col = 'Area'
linear_cols = ['Mean', 'Min', 'Max', 'Length']
results_pixels_df[area_col] = results_pixels_df[area_col] * sequential_ppu_squared
for col in linear_cols:
    results_pixels_df[col] = results_pixels_df[col] * sequential_ppu
results_pixels_df.to_csv('results_pixels.csv', index=False)
print("Fichier 'results_pixels.csv' généré.")


###########################################
# 2) CHARGEMENT ET TRONCATION DES MESURES RÉELLES (mm)
###########################################

df_real = pd.read_csv("mesures_reelles.csv")
df_real.columns = df_real.columns.str.strip()

# Troncature des données à la longueur minimale (pour gérer la désynchronisation des CSV)
# La longueur minimale est déduite de la taille réelle de df_real, qui cause l'erreur (25)
min_length = len(df_real) # Should be 25 based on the error.
print(f"Alignement sur la longueur minimale: {min_length}")

# Troncature des PPU
ppu_values = ppu_values[:min_length]

# Troncature des données nominales
distances_full = [
    3000,2900,2800,2700,2600,
    2500,2400,2300,2200,2100,
    2000,1900,1800,1700,1600,
    1500,1400,1300,1200,1100,
    1000,1000,
    900,900,
    800,800,
    700,700,
    600
]
distances = distances_full[:min_length]

# Calculs utiles
df_real["inv_p"] = 1 / df_real["Pavg_mm"].values
df_real["inv_q"] = 1 / df_real["Qavg_mm"].values

# CALCUL DES BARRES D'ERREUR RÉELLES (mm)
df_real["err_inv_q"] = df_real["Incertitude_mm"] / (df_real["Qavg_mm"] ** 2)
df_real["err_inv_p"] = df_real["Incertitude_mm"] / (df_real["Pavg_mm"] ** 2)

# Régression linéaire
a_real, b_real = np.polyfit(df_real["inv_q"], df_real["inv_p"], 1)
focale_reelle = 1 / b_real


###########################################
# 3) GRAPHIQUE DES MESURES RÉELLES (mm)
###########################################

plt.figure(figsize=(8,6))
plt.errorbar(
    df_real["inv_q"],
    df_real["inv_p"],
    xerr=df_real["err_inv_q"],
    yerr=df_real["err_inv_p"],
    fmt='o',
    capsize=4,
    label="Mesures réelles ± Incertitude"
)
x_line = np.linspace(df_real["inv_q"].min(), df_real["inv_q"].max(), 100)
y_line = a_real * x_line + b_real
plt.plot(x_line, y_line, label="Régression", linewidth=2, color='red')
plt.xlabel("1/q (mm⁻¹)")
plt.ylabel("1/p (mm⁻¹)")
plt.title("Mesures réelles – Loi des lentilles (avec barres d'erreur)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Graphiques/Manuel-mm-incertitudes.png")

print("\n--- RÉSULTATS RÉELS ---")
print(f"A = {a_real:.6f}")
print(f"B = {b_real:.6f}")
print(f"Focale estimée (réelle) : f = {focale_reelle:.2f} mm")

###########################################
# 4) CHARGEMENT ET CALCULS DES DONNÉES IMAGEJ (pixels)
###########################################

df_img_pixels = pd.read_csv("results_pixels.csv")
df_img_pixels.columns = df_img_pixels.columns.str.strip()

# Extraction et Troncature des p et q en pixels
p_values = df_img_pixels["Length"].iloc[0::2].reset_index(drop=True)
q_values = df_img_pixels["Length"].iloc[1::2].reset_index(drop=True)

p_values = p_values[:min_length] # Troncature
q_values = q_values[:min_length] # Troncature


df_img2 = pd.DataFrame({
    "distance_mm": distances, # Déjà tronqué
    "p_pixels": p_values,
    "q_pixels": q_values
})

df_img2["p_pixels"] = pd.to_numeric(df_img2["p_pixels"], errors='coerce')
df_img2["q_pixels"] = pd.to_numeric(df_img2["q_pixels"], errors='coerce')

# CALCUL DES BARRES D'ERREUR IMAGEJ (pixels)
# 1. Convertir Incertitude_mm en pixels en utilisant PPU séquentiel
uncertainty_mm_array = df_real["Incertitude_mm"].values
uncertainty_pixels = uncertainty_mm_array * np.array(ppu_values)

# 2. Appliquer la propagation de l'erreur aux valeurs en pixels
df_img2["err_inv_q"] = uncertainty_pixels / (df_img2["q_pixels"].values ** 2)
df_img2["err_inv_p"] = uncertainty_pixels / (df_img2["p_pixels"].values ** 2)

# Calculs (en pixels⁻¹)
df_img2["inv_p"] = 1 / df_img2["p_pixels"]
df_img2["inv_q"] = 1 / df_img2["q_pixels"]

# Régression
a_img, b_img = np.polyfit(df_img2["inv_q"], df_img2["inv_p"], 1)
focale_img_pixels = 1 / b_img

# Conversion de la focale de pixels à mm
focale_img_mm = focale_img_pixels * mm_per_pixel_factor


###########################################
# 5) GRAPHIQUE DES MESURES IMAGEJ (pixels)
###########################################

plt.figure(figsize=(8,6))
plt.errorbar(
    df_img2["inv_q"],
    df_img2["inv_p"],
    xerr=df_img2["err_inv_q"],
    yerr=df_img2["err_inv_p"],
    fmt='o',
    capsize=4,
    label="Mesures ImageJ ± Incertitude"
)

x_line2 = np.linspace(df_img2["inv_q"].min(), df_img2["inv_q"].max(), 100)
y_line2 = a_img * x_line2 + b_img
plt.plot(x_line2, y_line2, label="Régression", linewidth=2, color='red')

plt.xlabel("1/q (pixels⁻¹)")
plt.ylabel("1/p (pixels⁻¹)")
plt.title("Mesures ImageJ – Loi des lentilles (avec barres d'erreur)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Graphiques/ImageJ-Px-incertitudes.png")


print("\n--- RÉSULTATS IMAGEJ ---")
print(f"A = {a_img:.6f} (sans unité)")
print(f"B = {b_img:.6f} (pixels⁻¹)")
print(f"Focale estimée (ImageJ) : f = {focale_img_pixels:.2f} pixels")
print(f"Focale estimée (ImageJ) : f = {focale_img_mm:.2f} mm (Convertie à l'aide du PPU moyen)")