# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

#%%

folder = Path(".")
workbooks = sorted(folder.glob("FCC_Magnet_Report_*.xlsx"))


# %%
fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 9),
    sharex=True,
    constrained_layout=True,
)

for workbook in workbooks:

    df = pd.read_excel(
        workbook,
        sheet_name="Elements_Flat",
        header=1,
    )

    df.columns = [
        str(column).replace("\n", " ").strip()
        for column in df.columns
    ]

    dipoles = df[df["Type"] == "RBend"].copy()

    energy = workbook.stem.split("_")[-1].upper()

    dipoles["B-field [mT]"] = dipoles["B-field [T]"] * 1000

    print(f"{energy}: {len(dipoles)} dipoles")

    axes[0].scatter(
        dipoles["s [m]"],
        dipoles["Length [m]"],
        s=4,
        alpha=0.6,
        label=energy,
    )

    axes[1].scatter(
        dipoles["s [m]"],
        dipoles["B-field [mT]"],
        s=4,
        alpha=0.6,
        label=energy,
    )



axes[0].set_ylabel("Length [m]")


axes[1].set_xlabel("s [m]")
axes[1].set_ylabel("B-field [mT]")

for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(title="Energy")

plt.show()



Path("plots").mkdir(exist_ok=True)

fig.savefig(
    "plots/dipole_length_and_field_all_energies.png",
    dpi=180,
)
# %%
