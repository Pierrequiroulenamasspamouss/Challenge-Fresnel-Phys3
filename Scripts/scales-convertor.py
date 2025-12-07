import pandas as pd
import numpy as np

# --- 1. Load and Clean scales.csv ---
# Load with semicolon delimiter
scales_df = pd.read_csv('scales.csv', delimiter=';')
# Strip whitespace from column names
scales_df.columns = scales_df.columns.str.strip()
ppu_col_name = 'PPU (pixel-per-unit)'

# Clean the PPU column: replace ',' with '.' and convert to float
scales_df[ppu_col_name] = (
    scales_df[ppu_col_name]
    .astype(str)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

# Get the PPU values as a list. scales_df has 31 rows.
ppu_values_full = scales_df[ppu_col_name].tolist()

# --- 2. Load and Clean Results.csv ---
results_df = pd.read_csv('Results.csv')
# Strip whitespace from column names
results_df.columns = results_df.columns.str.strip()

# --- 3. Map PPU to Measurements (Sequential Mapping) ---
num_results = len(results_df) # 58
num_scales_needed = num_results // 2 # 58 / 2 = 29

# Slice the PPU values to use only the first 29 scales
ppu_values = ppu_values_full[:num_scales_needed]

# Repeat each PPU value twice to match the 58 measurements
sequential_ppu = np.repeat(ppu_values, 2)

# Double check the length (should be 58)
if len(sequential_ppu) != num_results:
    # This check is just for safety, should pass now.
    raise ValueError("Internal error: PPU array length still does not match results length.")

# Create the squared PPU array for Area conversion
sequential_ppu_squared = sequential_ppu ** 2

# --- 4. Perform Conversion ---
results_pixels_df = results_df.copy()

area_col = 'Area'
linear_cols = ['Mean', 'Min', 'Max', 'Length']

# Apply the squared PPU array to the Area column
results_pixels_df[area_col] = results_pixels_df[area_col] * sequential_ppu_squared

# Apply the linear PPU array to the linear distance columns
for col in linear_cols:
    results_pixels_df[col] = results_pixels_df[col] * sequential_ppu

# --- 5. Save Output ---
results_pixels_df.to_csv('results_pixels.csv', index=False)

print("Conversion séquentielle effectuée.")
print("\nResults Pixels DataFrame Head (après conversion séquentielle):")
print(results_pixels_df.head())