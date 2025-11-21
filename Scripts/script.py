import argparse
import os
import sys
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
script.py

Usage:
    python script.py --file data.xlsx [--sheet Sheet1] [--x XColumn] [--y Y1,Y2] [--out plot.png] [--show]

This script opens an Excel file, parses numeric columns using pandas + numpy,
and generates a graph using matplotlib. Numpy is used for numeric conversions
and simple statistics.
"""


import matplotlib.pyplot as plt


def parse_args():
        p = argparse.ArgumentParser(description="Plot Excel data (uses numpy/pandas/matplotlib).")
        p.add_argument("--file", "-f", required=True, help="Path to Excel file (.xlsx, .xls)")
        p.add_argument("--sheet", "-s", default=0,
                                     help="Sheet name or index (default: first sheet).")
        p.add_argument("--x", help="Column name or zero-based index to use for x-axis (default: first column).")
        p.add_argument("--y", help="Comma-separated column names or indices to plot as y (default: all other numeric columns).")
        p.add_argument("--out", "-o", default="plot.png", help="Output image file (PNG).")
        p.add_argument("--show", action="store_true", help="Show the plot interactively.")
        return p.parse_args()


def load_excel(path, sheet):
        if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
        # pandas will handle sheet as name or index
        df = pd.read_excel(path, sheet_name=sheet)
        return df


def pick_columns(df, sel):
        """
        sel: None or comma-separated list of names or indices
        Returns list of column labels
        """
        if sel is None:
                return None
        parts = [p.strip() for p in sel.split(",") if p.strip() != ""]
        cols = []
        for p in parts:
                # try integer index
                try:
                        idx = int(p)
                        cols.append(df.columns[idx])
                except Exception:
                        # treat as name
                        if p in df.columns:
                                cols.append(p)
                        else:
                                raise KeyError(f"Column '{p}' not found in sheet. Available: {list(df.columns)}")
        return cols


def to_numeric_series(s):
        # coerce non-numeric to NaN, then drop NaN
        return pd.to_numeric(s, errors="coerce")


def prepare_xy(df, x_sel, y_sel):
        # Determine x column
        if x_sel is None:
                x_col = df.columns[0]
        else:
                x_col = x_sel
        if isinstance(x_col, int):
                x_col = df.columns[x_col]

        # Determine y columns
        if y_sel is None:
                # all other columns that are numeric after coercion
                candidate_cols = [c for c in df.columns if c != x_col]
                numeric_cols = []
                for c in candidate_cols:
                        s = to_numeric_series(df[c])
                        if s.notna().any():
                                numeric_cols.append(c)
                y_cols = numeric_cols
        else:
                y_cols = y_sel

        if len(y_cols) == 0:
                raise ValueError("No y columns selected or found to plot.")

        # Convert to numpy arrays, align and drop rows with NaN in x or all y's NaN
        data = df.copy()
        data[x_col] = to_numeric_series(data[x_col])
        for c in y_cols:
                data[c] = to_numeric_series(data[c])

        # Drop rows where x is NaN
        data = data.dropna(subset=[x_col])
        # For convenience, drop rows where all selected y are NaN
        data = data.dropna(subset=y_cols, how="all")

        x = data[x_col].to_numpy(dtype=float)
        ys = {c: data[c].to_numpy(dtype=float) for c in y_cols}
        return x_col, y_cols, x, ys


def plot_data(filename, sheet, x_col, y_cols, x, ys, out_path, show):
        plt.figure(figsize=(8, 5))
        for c in y_cols:
                y = ys[c]
                # Some y arrays might contain NaN where x exists; mask them
                mask = ~np.isnan(y) & ~np.isnan(x)
                if mask.sum() == 0:
                        print(f"Warning: column '{c}' has no valid numeric points, skipping.", file=sys.stderr)
                        continue
                x_masked = x[mask]
                y_masked = y[mask]
                mean = np.nanmean(y_masked)
                std = np.nanstd(y_masked)
                plt.plot(x_masked, y_masked, marker='o', linestyle='-', label=f"{c} (μ={mean:.3g}, σ={std:.3g})")

        plt.xlabel(str(x_col))
        plt.ylabel("Value")
        title_sheet = f" - sheet: {sheet}" if sheet != 0 and sheet is not None else ""
        plt.title(f"{os.path.basename(filename)}{title_sheet}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        if show:
                plt.show()
        plt.close()


def main():
        args = parse_args()
        try:
                df = load_excel(args.file, args.sheet)
        except Exception as e:
                print(f"Error loading Excel file: {e}", file=sys.stderr)
                sys.exit(1)

        try:
                x_sel = None
                y_sel = None
                if args.x:
                        # convert to column label (int index or name)
                        try:
                                xi = int(args.x)
                                x_sel = df.columns[xi]
                        except Exception:
                                if args.x in df.columns:
                                        x_sel = args.x
                                else:
                                        print(f"X column '{args.x}' not found.", file=sys.stderr)
                                        sys.exit(1)
                if args.y:
                        y_sel = pick_columns(df, args.y)

                x_col, y_cols, x, ys = prepare_xy(df, x_sel, y_sel)
        except Exception as e:
                print(f"Error preparing data: {e}", file=sys.stderr)
                sys.exit(1)

        try:
                plot_data(args.file, args.sheet, x_col, y_cols, x, ys, args.out, args.show)
        except Exception as e:
                print(f"Error plotting data: {e}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
        main()