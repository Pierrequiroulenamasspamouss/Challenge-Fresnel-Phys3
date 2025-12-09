import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =================CONFIGURATION=================
NOM_FICHIER = 'CHAAF15.txt' 
# ===============================================

def analyser_fichier_chaaf(filename):
    print(f"--- Lecture du fichier : {filename} ---")
    
    if not os.path.exists(filename):
        print(f"ERREUR : Le fichier '{filename}' est introuvable.")
        return

    try:
        # Lecture et nettoyage
        df = pd.read_csv(filename, sep=';', comment='#', header=None, engine='python')
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # Gestion dynamique des colonnes
        if df.shape[1] >= 5:
            df = df.iloc[:, :6]
            cols = ['p_px', 'dp_px', 'q_px', 'dq_px', 'scale', 'dscale']
            df.columns = cols[:df.shape[1]]
        else:
            print("Erreur : Pas assez de colonnes.")
            return

        # Conversion string -> float
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # --- CALCULS PHYSIQUES ---
        
        # 1. Conversion Pixels -> mm
        df["p_mm"] = df["p_px"] / df["scale"]
        df["q_mm"] = df["q_px"] / df["scale"]

        # 2. Propagation incertitudes (p et q)
        df["dp_mm"] = df["dp_px"] / df["scale"]
        df["dq_mm"] = df["dq_px"] / df["scale"]

        # 3. Inverses
        df["inv_p"] = 1 / df["p_mm"]
        df["inv_q"] = 1 / df["q_mm"]

        # 4. Incertitudes inverses
        df["d_inv_p"] = df["dp_mm"] / (df["p_mm"]**2)
        df["d_inv_q"] = df["dq_mm"] / (df["q_mm"]**2)

        # --- NOUVEAU : CALCUL DE LA FOCALE ---
        
        # Méthode A : Régression Linéaire (1/p = a * 1/q + b) => b = 1/f
        coeffs = np.polyfit(df["inv_q"], df["inv_p"], 1)
        pente = coeffs[0]
        ordonnee_origine = coeffs[1]
        
        # Calcul de f via l'ordonnée à l'origine
        f_reg = 1 / ordonnee_origine
        
        # Méthode B : Moyenne statistique directe
        # f = (p*q)/(p+q)
        vals_f = (df["p_mm"] * df["q_mm"]) / (df["p_mm"] + df["q_mm"])
        f_moy = vals_f.mean()
        f_std = vals_f.std() # Écart-type

        print("\n" + "="*30)
        print(" RÉSULTATS DU CALCUL DE FOCALE")
        print("="*30)
        print(f"Méthode 1 (Régression) f  = {f_reg:.2f} mm")
        print(f"Méthode 2 (Moyenne)    f  = {f_moy:.2f} mm +/- {f_std:.2f} mm")
        print(f"Pente de la droite        = {pente:.3f} (théorique : -1.0)")
        print("="*30 + "\n")

        # --- GRAPHIQUE ---
        plt.figure(figsize=(10, 7))

        plt.errorbar(df["inv_q"], df["inv_p"], 
                     xerr=df["d_inv_q"], yerr=df["d_inv_p"], 
                     fmt='o', color='blue', ecolor='red', capsize=3, 
                     label='Mesures')

        poly_fn = np.poly1d(coeffs)
        x_range = np.linspace(df["inv_q"].min(), df["inv_q"].max(), 100)
        
        # Affichage du résultat dans la légende
        label_reg = f'Fit: y={pente:.2f}x + {ordonnee_origine:.3f}\nFocale calc. = {f_reg:.1f} mm'
        plt.plot(x_range, poly_fn(x_range), 'k--', alpha=0.8, label=label_reg)

        #plt.title(f"Détermination de la focale (F = {f_reg:.1f} mm)")
        plt.xlabel(r"$1/q$ [$mm^{-1}$]")
        plt.ylabel(r"$1/p$ [$mm^{-1}$]")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{filename.replace(".txt", "")}.png')
        plt.show()

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    analyser_fichier_chaaf(NOM_FICHIER)