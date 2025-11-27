import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################################
# 1) CHARGEMENT DES MESURES RÉELLES
###########################################

df_real = pd.read_csv("mesures_reelles.csv")

# Calculs utiles
df_real["inv_p"] = 1 / df_real["Pavg_mm"].values
df_real["inv_q"] = 1 / df_real["Qavg_mm"].values

# Régression linéaire : inv_p = a * inv_q + b
a_real, b_real = np.polyfit(df_real["inv_q"], df_real["inv_p"], 1)
focale_reelle = 1 / b_real

###########################################
# 2) GRAPHIQUE DES MESURES RÉELLES
###########################################

plt.figure(figsize=(8,6))
plt.scatter(df_real["inv_q"], df_real["inv_p"], label="Mesures réelles")

x_line = np.linspace(df_real["inv_q"].min(), df_real["inv_q"].max(), 100)
y_line = a_real * x_line + b_real
plt.plot(x_line, y_line, label="Régression", linewidth=2)

plt.xlabel("1/q (mm⁻¹)")
plt.ylabel("1/p (mm⁻¹)")
plt.title("Mesures réelles – Loi des lentilles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- RÉSULTATS RÉELS ---")
print(f"A = {a_real:.6f}")
print(f"B = {b_real:.6f}")
print(f"Focale estimée (réelle) : f = {focale_reelle:.2f} mm")


###########################################
# 3) CHARGEMENT DES DONNÉES IMAGEJ
###########################################

df_img = pd.read_csv("Results.csv")

# Pour rappel : alternance dans Results.csv :
#   ligne impaire (index pair) = données p
#   ligne paire (index impair) = données q
# Exemple :
#   3000mm → p = row0.length , q = row1.length
#   2900mm → p = row2.length , q = row3.length
#   ...
#   1000mm → p = row? , q = row?
#   900mm → ...
#   600mm → 2 images → donc 4 lignes : p1,q1,p2,q2

# On fabrique un tableau propre p/q
p_values = df_img["Length"].iloc[0::2].reset_index(drop=True)
q_values = df_img["Length"].iloc[1::2].reset_index(drop=True)

# Distances nominales
distances = [
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

df_img2 = pd.DataFrame({
    "distance_mm": distances[:len(p_values)],
    "p_mm": p_values,
    "q_mm": q_values
})
df_img2["p_mm"] = pd.to_numeric(df_img2["p_mm"], errors='coerce')
df_img2["q_mm"] = pd.to_numeric(df_img2["q_mm"], errors='coerce')
# Calculs
df_img2["inv_p"] = 1 / df_img2["p_mm"]
df_img2["inv_q"] = 1 / df_img2["q_mm"]

# Régression
a_img, b_img = np.polyfit(df_img2["inv_q"], df_img2["inv_p"], 1)
focale_img = 1 / b_img


###########################################
# 4) GRAPHIQUE DES MESURES IMAGEJ
###########################################

plt.figure(figsize=(8,6))
plt.scatter(df_img2["inv_q"], df_img2["inv_p"], label="Mesures ImageJ")

x_line2 = np.linspace(df_img2["inv_q"].min(), df_img2["inv_q"].max(), 100)
y_line2 = a_img * x_line2 + b_img
plt.plot(x_line2, y_line2, label="Régression", linewidth=2)

plt.xlabel("1/q (mm⁻¹)")
plt.ylabel("1/p (mm⁻¹)")
plt.title("Mesures ImageJ – Loi des lentilles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- RÉSULTATS IMAGEJ ---")
print(f"A = {a_img:.6f}")
print(f"B = {b_img:.6f}")
print(f"Focale estimée (ImageJ) : f = {focale_img:.2f} mm")




"""
# mesures reelles osef 600,274.5,300,325.5,51

#ImageJ



59,289.479,125.838,54.721,253.311,1.878,229.659
60,511.308,153.811,32.148,251.975,-177.879,406.670
    
61,132.357,132.309,6.506,253.889,-1.532,272.347
62,231.271,110.164,5.000,252.667,-178.248,476.296

"""
