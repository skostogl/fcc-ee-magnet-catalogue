#!/usr/bin/env python3
"""Example cross-energy plots from the generated FCC-ee Excel catalogues."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_dipoles(workbook_path: Path) -> pd.DataFrame:
    """Read the Elements_Flat sheet and return only physical dipole rows."""
    df = pd.read_excel(workbook_path, sheet_name="Elements_Flat", header=1)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    dipoles = df[df["Type"] == "RBend"].copy()
    dipoles["Energy"] = workbook_path.stem.rsplit("_", 1)[-1].upper()
    dipoles["B-field [mT]"] = pd.to_numeric(
        dipoles["B-field [T]"], errors="coerce"
    ) * 1000.0
    dipoles["Length [m]"] = pd.to_numeric(dipoles["Length [m]"], errors="coerce")
    dipoles["s [m]"] = pd.to_numeric(dipoles["s [m]"], errors="coerce")
    return dipoles.dropna(subset=["s [m]", "Length [m]", "B-field [mT]"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dipole length and field for every generated energy."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing FCC_Magnet_Report_<energy>.xlsx files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "plots"
        / "dipole_length_and_field_all_energies.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    workbooks = sorted(args.input_dir.glob("FCC_Magnet_Report_*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(
            f"No FCC_Magnet_Report_*.xlsx files found in {args.input_dir}"
        )

    datasets = [(path, load_dipoles(path)) for path in workbooks]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)

    for path, dipoles in datasets:
        energy = dipoles["Energy"].iloc[0]
        axes[0].scatter(
            dipoles["s [m]"],
            dipoles["Length [m]"],
            s=4,
            alpha=0.55,
            label=energy,
        )
        axes[1].scatter(
            dipoles["s [m]"],
            dipoles["B-field [mT]"],
            s=4,
            alpha=0.55,
            label=energy,
        )
        print(f"{energy}: {len(dipoles)} dipoles read from {path.name}")

    axes[0].set_title("FCC-ee dipole length by longitudinal position")
    axes[0].set_ylabel("Dipole length [m]")
    axes[1].set_title("FCC-ee dipole field by longitudinal position")
    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("B-field [mT]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(title="Energy")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
