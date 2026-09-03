# %%
from pathlib import Path
import re

import numpy as np
import pandas as pd
import xtrack as xt
import matplotlib.pyplot as plt


# Change only this line when checking another energy.
energy = "z"

project_dir = Path(__file__).resolve().parent
workbook = project_dir / f"FCC_Magnet_Report_{energy}.xlsx"
lattice_b1 = (
    project_dir
    / "acc-models-fcc-ee"
    / "lattices"
    / energy
    / f"fccee_{energy}_beam_1_positrons.json"
)
lattice_b2 = (
    project_dir
    / "acc-models-fcc-ee"
    / "lattices"
    / energy
    / f"fccee_{energy}_beam_2_electrons.json"
)


# %%
# Read only the flat list. This is much faster than rebuilding the catalogue.
flat = pd.read_excel(workbook, sheet_name="Elements_Flat", header=1)
flat.columns = [re.sub(r"\s+", " ", str(column)).strip() for column in flat.columns]
flat = flat.rename(columns={
    "B1 s [m]": "s [m]",
    "B2 element name": "Closest B2 element",
    "Centre horizontal separation [m]": "Horizontal separation [m]",
    "Centre vertical separation [m]": "Vertical separation [m]",
})
flat = flat[flat["Type"].isin(["RBend", "Quadrupole", "Sextupole"])].copy()

required_columns = [
    "Old Name",
    "Type",
    "Region",
    "s [m]",
    "Closest B2 element",
    "Horizontal separation [m]",
    "Vertical separation [m]",
]

missing_columns = [column for column in required_columns if column not in flat.columns]
if missing_columns:
    raise KeyError(f"Missing Elements_Flat columns: {missing_columns}")


# %%
# 1. Rows rejected by the production matching have no B2 element name.
has_b2_match = flat["Closest B2 element"].notna() & flat["Closest B2 element"].ne("")
rejected = flat[~has_b2_match].copy()


# %%
# 2. B2 elements assigned to more than one accepted B1 magnet.
accepted = flat[has_b2_match].copy()
duplicate_mask = accepted["Closest B2 element"].duplicated(keep=False)
duplicate_assignments = accepted[duplicate_mask].sort_values(
    ["Closest B2 element", "s [m]"]
)

duplicate_summary = (
    duplicate_assignments.groupby("Closest B2 element", as_index=False)
    .agg(
        Number_of_B1_magnets=("Old Name", "size"),
        B1_magnets=("Old Name", lambda values: ", ".join(values.astype(str))),
        B1_regions=("Region", lambda values: ", ".join(sorted(set(values.astype(str))))),
    )
    .sort_values(["Number_of_B1_magnets", "Closest B2 element"], ascending=[False, True])
)


# %%
# 3. Actual B2 magnets that were never used by an accepted match.
loaded_b2 = xt.load(str(lattice_b2))
line_b2_check = loaded_b2.fccee_e_ring
table_b2 = line_b2_check.get_table()

b2_magnets = pd.DataFrame({
    "B2 element": np.asarray(table_b2.name).astype(str),
    "B2 type": np.asarray(table_b2.element_type).astype(str),
    "B2 s [m]": np.asarray(table_b2.s, dtype=float),
})
b2_magnets = b2_magnets[
    b2_magnets["B2 type"].isin(["RBend", "Quadrupole", "Sextupole"])
].copy()

used_b2_names = set(accepted["Closest B2 element"].dropna().astype(str))
unused_b2_magnets = b2_magnets[
    ~b2_magnets["B2 element"].isin(used_b2_names)
].sort_values("B2 s [m]")


# %%
# 4. Recalculate the accepted pairs at the centres of both magnets.
loaded_b1 = xt.load(str(lattice_b1))
line_b1_check = loaded_b1.fccee_p_ring
table_b1 = line_b1_check.get_table()

half_crossing_angle = 15e-3
survey_b1 = line_b1_check.survey(theta0=half_crossing_angle)
survey_b2 = line_b2_check.survey(theta0=np.pi - half_crossing_angle)


def survey_arrays(survey):
    s = np.asarray(survey["s"], dtype=float)
    xyz = np.column_stack([
        np.asarray(survey["X"], dtype=float),
        np.asarray(survey["Y"], dtype=float),
        np.asarray(survey["Z"], dtype=float),
    ])
    unique_s, unique_index = np.unique(s, return_index=True)
    return unique_s, xyz[unique_index]


def element_positions(line, table):
    positions = {}
    circumference = float(np.asarray(table.s, dtype=float)[-1])
    for name, s in zip(table.name, table.s):
        name = str(name)
        if name in positions:
            continue
        element = line.element_dict.get(name)
        length = float(getattr(element, "length", 0.0) or 0.0)
        start = float(s)
        end = start + length
        centre = (start + 0.5 * length) % circumference
        positions[name] = (start, centre, end)
    return positions


s_survey_b1, xyz_survey_b1 = survey_arrays(survey_b1)
s_survey_b2, xyz_survey_b2 = survey_arrays(survey_b2)
positions_b1 = element_positions(line_b1_check, table_b1)
positions_b2 = element_positions(line_b2_check, table_b2)


def xyz_at_s(s, survey_s, survey_xyz):
    return np.array([
        np.interp(s, survey_s, survey_xyz[:, axis])
        for axis in range(3)
    ])


centre_rows = []
for _, row in accepted.iterrows():
    name_b1 = str(row["Old Name"])
    name_b2 = str(row["Closest B2 element"])

    if name_b1 not in positions_b1 or name_b2 not in positions_b2:
        continue

    b1_start, b1_centre, b1_end = positions_b1[name_b1]
    b2_start, b2_centre, b2_end = positions_b2[name_b2]

    xyz_b1 = xyz_at_s(b1_centre, s_survey_b1, xyz_survey_b1)
    xyz_b2 = xyz_at_s(b2_centre, s_survey_b2, xyz_survey_b2)
    delta = xyz_b2 - xyz_b1

    centre_rows.append({
        **row.to_dict(),
        "B1 s start [m]": b1_start,
        "B1 s centre [m]": b1_centre,
        "B1 s end [m]": b1_end,
        "B2 s start [m]": b2_start,
        "B2 s centre [m]": b2_centre,
        "B2 s end [m]": b2_end,
        "Centre horizontal separation [m]": float(np.hypot(delta[0], delta[2])),
        "Centre vertical separation [m]": float(abs(delta[1])),
        "Centre 3D inter-beam distance [m]": float(np.linalg.norm(delta)),
    })

centre_comparison = pd.DataFrame(centre_rows)


# %%
# 5. Check that every accepted pair has both centre-distance components.
missing_values = accepted[
    accepted[
        [
            "Closest B2 element",
            "Horizontal separation [m]",
            "Vertical separation [m]",
        ]
    ].isna().any(axis=1)
].copy()


# %%
print("\nBEAM-2 MATCH CHECK")
print("=" * 72)
print(f"Workbook: {workbook.name}")
print(f"Accepted B1 rows:            {len(accepted):6d}")
print(f"Rejected/unmatched B1 rows:  {len(rejected):6d}")
print(f"Duplicated B2 element names: {len(duplicate_summary):6d}")
print(f"B1 rows in duplicate groups: {len(duplicate_assignments):6d}")
print(f"Unused B2 magnets:           {len(unused_b2_magnets):6d}")
print(f"Missing accepted values:     {len(missing_values):6d}")

if len(duplicate_summary):
    print("\nFIRST DUPLICATE ASSIGNMENTS")
    print(duplicate_summary.head(20).to_string(index=False))

if len(rejected):
    print("\nFIRST REJECTED ROWS")
    print(
        rejected[
            [
                "Old Name",
                "Type",
                "Region",
                "Closest B2 element",
            ]
        ].head(20).to_string(index=False)
    )


# %%
# Save compact diagnostic tables for filtering in Excel.
output_folder = project_dir / "match_checks"
output_folder.mkdir(exist_ok=True)

duplicate_assignments.to_csv(
    output_folder / f"duplicate_b2_assignments_{energy}.csv",
    index=False,
)
unused_b2_magnets.to_csv(
    output_folder / f"unused_b2_magnets_{energy}.csv",
    index=False,
)
rejected.to_csv(
    output_folder / f"rejected_b2_matches_{energy}.csv",
    index=False,
)
centre_comparison.to_csv(
    output_folder / f"centre_distance_comparison_{energy}.csv",
    index=False,
)

print(f"\nDiagnostic CSV files saved in: {output_folder}")


# %%
# Focus on one suspicious B2 element and show neighbouring assignments.
element_to_check = "mbd.6662"#"qd0ar.0" #mbd.6649"


rows_to_check = accepted[
    accepted["Closest B2 element"].eq(element_to_check)
]

if len(rows_to_check):
    first_index = rows_to_check.index.min()
    context_names = accepted.loc[
        max(accepted.index.min(), first_index - 3): first_index + 30,
        "Old Name",
    ]
    context = centre_comparison[
        centre_comparison["Old Name"].isin(context_names)
    ]
    print(f"\nCONTEXT AROUND {element_to_check}")
    print(
        context[
            [
                "Old Name",
                "Region",
                "s [m]",
                "Closest B2 element",
                "B1 s start [m]",
                "B1 s centre [m]",
                "B1 s end [m]",
                "B2 s start [m]",
                "B2 s centre [m]",
                "B2 s end [m]",
                "Centre horizontal separation [m]",
                "Centre vertical separation [m]",
                "Centre 3D inter-beam distance [m]",
            ]
        ].to_string(index=False)
    )

    # %%
    import matplotlib.pyplot as plt
    # Local horizontal layout around the selected B1/B2 magnet pairs.
    margin = 10.0  # metres before/after the displayed magnet group
    b1_plot_start = context["B1 s start [m]"].min() - margin
    b1_plot_end = context["B1 s end [m]"].max() + margin
    b2_plot_start = context["B2 s start [m]"].min() - margin
    b2_plot_end = context["B2 s end [m]"].max() + margin

    def xyz_at_many_s(s_values, survey_s, survey_xyz):
        s_values = np.asarray(s_values, dtype=float)
        return np.column_stack([
            np.interp(s_values, survey_s, survey_xyz[:, axis])
            for axis in range(3)
        ])

    b1_plot_s = np.linspace(b1_plot_start, b1_plot_end, 2000)
    b2_plot_s = np.linspace(b2_plot_start, b2_plot_end, 2000)
    b1_plot_xyz = xyz_at_many_s(b1_plot_s, s_survey_b1, xyz_survey_b1)
    b2_plot_xyz = xyz_at_many_s(b2_plot_s, s_survey_b2, xyz_survey_b2)

    # Local frame: longitudinal direction follows B1; horizontal is the
    # perpendicular direction in the global floor plane.
    origin = xyz_at_s(
        context["B1 s centre [m]"].mean(), s_survey_b1, xyz_survey_b1
    )
    longitudinal = b1_plot_xyz[-1] - b1_plot_xyz[0]
    longitudinal /= np.linalg.norm(longitudinal)
    global_up = np.array([0.0, 1.0, 0.0])
    horizontal_local = np.cross(global_up, longitudinal)
    horizontal_local /= np.linalg.norm(horizontal_local)

    def local_coordinates(xyz):
        relative = np.asarray(xyz) - origin
        return relative @ longitudinal, relative @ horizontal_local

    b1_long, b1_horizontal = local_coordinates(b1_plot_xyz)
    b2_long, b2_horizontal = local_coordinates(b2_plot_xyz)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(b1_long, b1_horizontal, color="tab:blue", lw=1.2, label="B1 trajectory")
    ax.plot(b2_long, b2_horizontal, color="tab:red", lw=1.2, label="B2 trajectory")

    for number, (_, pair) in enumerate(context.iterrows()):
        b1_span_s = np.linspace(pair["B1 s start [m]"], pair["B1 s end [m]"], 80)
        b2_span_s = np.linspace(pair["B2 s start [m]"], pair["B2 s end [m]"], 80)
        b1_span_xyz = xyz_at_many_s(b1_span_s, s_survey_b1, xyz_survey_b1)
        b2_span_xyz = xyz_at_many_s(b2_span_s, s_survey_b2, xyz_survey_b2)
        b1_span_long, b1_span_horizontal = local_coordinates(b1_span_xyz)
        b2_span_long, b2_span_horizontal = local_coordinates(b2_span_xyz)

        ax.plot(b1_span_long, b1_span_horizontal, color="tab:blue", lw=7, solid_capstyle="butt")
        ax.plot(b2_span_long, b2_span_horizontal, color="tab:red", lw=7, solid_capstyle="butt")

        b1_centre_xyz = xyz_at_s(
            pair["B1 s centre [m]"], s_survey_b1, xyz_survey_b1
        )
        b2_centre_xyz = xyz_at_s(
            pair["B2 s centre [m]"], s_survey_b2, xyz_survey_b2
        )
        b1_centre_long, b1_centre_horizontal = local_coordinates(b1_centre_xyz)
        b2_centre_long, b2_centre_horizontal = local_coordinates(b2_centre_xyz)

        ax.plot(
            [b1_centre_long, b2_centre_long],
            [b1_centre_horizontal, b2_centre_horizontal],
            color="0.55",
            lw=0.8,
            ls="--",
            zorder=1,
        )
        label_shift = 8 + 7 * (number % 2)
        ax.annotate(
            pair["Old Name"],
            (b1_centre_long, b1_centre_horizontal),
            xytext=(0, -label_shift),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="tab:blue",
            rotation=30,
        )
        ax.annotate(
            pair["Closest B2 element"],
            (b2_centre_long, b2_centre_horizontal),
            xytext=(0, label_shift),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:red",
            rotation=30,
        )
        # annotate the distance between B1 and B2 centres
        distance = np.linalg.norm(b2_centre_xyz - b1_centre_xyz)
        ax.annotate(
            f"{distance:.2f} m",
            ((b1_centre_long + b2_centre_long) / 2, (b1_centre_horizontal + b2_centre_horizontal) / 2),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.55",
        )

    ax.set_xlabel("Local longitudinal position [m]")
    ax.set_ylabel("Local horizontal position [m]")
    ax.set_title(f"B1/B2 layout around {element_to_check}")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.margins(x=0.02, y=0.30)
    fig.tight_layout()

    layout_plot = output_folder / f"layout_around_{element_to_check.replace('.', '_')}_{energy}.png"
    fig.savefig(layout_plot, dpi=180)
    print(f"\nLocal layout plot saved to: {layout_plot}")
    plt.show()

# %%
# Plot every machine region containing an inter-beam horizontal separation
# clearly above the nominal 35 cm.  A 1 mm tolerance avoids selecting normal
# arc magnets because of rounding/interpolation noise around exactly 0.35 m.
horizontal_limit = 0.35
horizontal_tolerance = 1e-3
selection_limit = horizontal_limit + horizontal_tolerance
maximum_labels_per_plot = 1200000000

# Reopen the catalogue so this check tests what was actually written to Excel.
flat_for_region_plots = pd.read_excel(
    workbook,
    sheet_name="Elements_Flat",
    header=1,
)
flat_for_region_plots.columns = [
    str(column).replace("\n", " ").strip()
    for column in flat_for_region_plots.columns
]

required_plot_columns = [
    "Old Name",
    "Region",
    "B2 element name",
    "Centre horizontal separation [m]",
]
missing_plot_columns = [
    column for column in required_plot_columns
    if column not in flat_for_region_plots.columns
]
if missing_plot_columns:
    raise KeyError(
        "Cannot create regional separation plots. Missing Elements_Flat "
        f"columns: {missing_plot_columns}"
    )

flat_for_region_plots["Centre horizontal separation [m]"] = pd.to_numeric(
    flat_for_region_plots["Centre horizontal separation [m]"],
    errors="coerce",
)
above_limit_in_excel = flat_for_region_plots[
    flat_for_region_plots["B2 element name"].notna()
    & (
        flat_for_region_plots["Centre horizontal separation [m]"]
        > selection_limit
    )
].copy()

regions_over_limit = (
    above_limit_in_excel.groupby("Region", dropna=True)
    .agg(
        elements_over_limit=("Old Name", "size"),
        maximum_horizontal_separation=(
            "Centre horizontal separation [m]",
            "max",
        ),
    )
    .sort_values("maximum_horizontal_separation", ascending=False)
)

regional_plot_folder = output_folder / "layout_regions_over_035m"
regional_plot_folder.mkdir(parents=True, exist_ok=True)


def xyz_at_many_s_region_plot(s_values, survey_s, survey_xyz):
    s_values = np.asarray(s_values, dtype=float)
    return np.column_stack([
        np.interp(s_values, survey_s, survey_xyz[:, axis])
        for axis in range(3)
    ])


regional_plot_records = []
for region, region_summary in regions_over_limit.iterrows():
    # Use the centre-based comparison table for the plotted magnet extents and
    # coordinates, but select the rows from the values read back from Excel.
    names_over_limit = set(
        above_limit_in_excel.loc[
            above_limit_in_excel["Region"].eq(region), "Old Name"
        ]
    )
    region_pairs = centre_comparison[
        centre_comparison["Region"].eq(region)
        & centre_comparison["Old Name"].isin(names_over_limit)
    ].copy()
    if region_pairs.empty:
        continue

    region_pairs = region_pairs.sort_values("B1 s centre [m]")
    b1_plot_s = region_pairs["B1 s centre [m]"].to_numpy()
    b1_plot_xyz = xyz_at_many_s_region_plot(
        b1_plot_s, s_survey_b1, xyz_survey_b1
    )
    # B2 runs in the opposite longitudinal direction and can cross s=0 inside
    # a B1 region.  Follow the matched B2 centres in B1 order instead of
    # interpolating from min(B2 s) to max(B2 s), which could draw the full ring.
    b2_plot_s = region_pairs["B2 s centre [m]"].to_numpy()
    b2_plot_xyz = xyz_at_many_s_region_plot(
        b2_plot_s, s_survey_b2, xyz_survey_b2
    )

    # Some regions (notably INS1) cross the s=0/circumference boundary.  Use
    # the best-fit direction of their physical points, so neither beam is
    # accidentally drawn once around the complete ring.
    origin = b1_plot_xyz.mean(axis=0)
    floor_offsets = b1_plot_xyz - origin
    floor_offsets[:, 1] = 0.0
    _, _, principal_axes = np.linalg.svd(floor_offsets, full_matrices=False)
    longitudinal = principal_axes[0]
    longitudinal[1] = 0.0
    longitudinal /= np.linalg.norm(longitudinal)
    global_up = np.array([0.0, 1.0, 0.0])
    horizontal_local = np.cross(global_up, longitudinal)
    horizontal_local /= np.linalg.norm(horizontal_local)

    def regional_local_coordinates(xyz):
        relative = np.asarray(xyz) - origin
        return relative @ longitudinal, relative @ horizontal_local

    b1_long, b1_horizontal = regional_local_coordinates(b1_plot_xyz)
    b2_long, b2_horizontal = regional_local_coordinates(b2_plot_xyz)
    longitudinal_order = np.argsort(b1_long)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        b1_long[longitudinal_order], b1_horizontal[longitudinal_order],
        color="tab:blue", lw=1.4, label="B1 trajectory",
    )
    ax.plot(
        b2_long[longitudinal_order], b2_horizontal[longitudinal_order],
        color="tab:red", lw=1.4, label="B2 trajectory",
    )

    # Draw every magnet pair above the limit.  Only label the largest cases,
    # otherwise long insertions become unreadable.
    rows_to_label = set(
        region_pairs.nlargest(
            maximum_labels_per_plot,
            "Centre horizontal separation [m]",
        ).index
    )
    for number, (pair_index, pair) in enumerate(region_pairs.iterrows()):
        b1_span_s = np.linspace(
            pair["B1 s start [m]"], pair["B1 s end [m]"], 80
        )
        b2_span_s = np.linspace(
            pair["B2 s start [m]"], pair["B2 s end [m]"], 80
        )
        b1_span_xyz = xyz_at_many_s_region_plot(
            b1_span_s, s_survey_b1, xyz_survey_b1
        )
        b2_span_xyz = xyz_at_many_s_region_plot(
            b2_span_s, s_survey_b2, xyz_survey_b2
        )
        b1_span_long, b1_span_horizontal = regional_local_coordinates(b1_span_xyz)
        b2_span_long, b2_span_horizontal = regional_local_coordinates(b2_span_xyz)
        ax.plot(
            b1_span_long, b1_span_horizontal,
            color="tab:blue", lw=5, solid_capstyle="butt", alpha=0.75,
        )
        ax.plot(
            b2_span_long, b2_span_horizontal,
            color="tab:red", lw=5, solid_capstyle="butt", alpha=0.75,
        )

        b1_centre_xyz = xyz_at_s(
            pair["B1 s centre [m]"], s_survey_b1, xyz_survey_b1
        )
        b2_centre_xyz = xyz_at_s(
            pair["B2 s centre [m]"], s_survey_b2, xyz_survey_b2
        )
        b1_centre_long, b1_centre_horizontal = regional_local_coordinates(
            b1_centre_xyz
        )
        b2_centre_long, b2_centre_horizontal = regional_local_coordinates(
            b2_centre_xyz
        )
        ax.plot(
            [b1_centre_long, b2_centre_long],
            [b1_centre_horizontal, b2_centre_horizontal],
            color="0.55", lw=0.7, ls="--", alpha=0.7, zorder=1,
        )

        if pair_index in rows_to_label:
            label_shift = 8 + 7 * (number % 2)
            ax.annotate(
                pair["Old Name"],
                (b1_centre_long, b1_centre_horizontal),
                xytext=(0, -label_shift), textcoords="offset points",
                ha="center", va="top", fontsize=7,
                color="tab:blue", rotation=30,
            )
            ax.annotate(
                pair["Closest B2 element"],
                (b2_centre_long, b2_centre_horizontal),
                xytext=(0, label_shift), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
                color="tab:red", rotation=30,
            )
            ax.annotate(
                f'{pair["Centre horizontal separation [m]"]:.2f} m',
                (
                    (b1_centre_long + b2_centre_long) / 2,
                    (b1_centre_horizontal + b2_centre_horizontal) / 2,
                ),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color="0.35",
            )

    maximum_distance = region_pairs[
        "Centre horizontal separation [m]"
    ].max()
    ax.set_xlabel("Local longitudinal position [m]")
    ax.set_ylabel("Local horizontal position [m]")
    ax.set_title(
        f"B1/B2 layout in {region}: "
        f"{len(region_pairs)} elements above {selection_limit:.3f} m "
        f"(maximum {maximum_distance:.3f} m)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.margins(x=0.02, y=0.18)
    fig.tight_layout()

    safe_region = str(region).replace("/", "_").replace(" ", "_")
    regional_plot = (
        regional_plot_folder
        / f"layout_{safe_region}_over_035m_{energy}.png"
    )
    fig.savefig(regional_plot, dpi=180)
    plt.close(fig)
    regional_plot_records.append({
        "Region": region,
        "Elements above limit": len(region_pairs),
        "Maximum horizontal separation [m]": maximum_distance,
        "Plot": str(regional_plot),
    })

regional_plot_summary = pd.DataFrame(regional_plot_records)
print(
    f"\nREGIONS WITH CENTRE HORIZONTAL SEPARATION > "
    f"{selection_limit:.3f} m"
)
if regional_plot_summary.empty:
    print("None found.")
else:
    print(regional_plot_summary.to_string(index=False))
    print(f"\nRegional layout plots saved in: {regional_plot_folder}")

plt.show()

# %

# %%
# %%
# Add regional B1/B2 layout plots beside Elements_Flat

from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter

workbook_path = project_dir / f"FCC_Magnet_Report_{energy}.xlsx"
plot_folder = (
    project_dir
    / "match_checks"
    / "layout_regions_over_035m"
)

wb = load_workbook(workbook_path)
ws = wb["Elements_Flat"]

# This script can be run repeatedly while checking the matching. Remove the
# previously embedded regional plots before adding the freshly generated set.
ws._images = []

# Find the Region column.
header_row = 2
headers = {
    ws.cell(header_row, column).value: column
    for column in range(1, ws.max_column + 1)
}

region_column = headers["Region"]

# Put plots two columns after the existing table.
plot_column = ws.max_column + 2
plot_column_letter = get_column_letter(plot_column)

# First Excel row occupied by each region.
first_row_by_region = {}

for row in range(header_row + 1, ws.max_row + 1):
    region = ws.cell(row, region_column).value

    if region and region not in first_row_by_region:
        first_row_by_region[region] = row

for region, row in first_row_by_region.items():
    plot_path = (
        plot_folder
        / f"layout_{region}_over_035m_{energy}.png"
    )

    if not plot_path.exists():
        continue

    image = Image(plot_path)
    image.width = 1600
    image.height = 640

    image.anchor = f"{plot_column_letter}{row}"
    ws.add_image(image)

    print(f"Added {region} plot at {plot_column_letter}{row}")

wb.save(workbook_path)
print(f"Plots added to {workbook_path.name}")
# %%

# plot

line   = xt.load("acc-models-fcc-ee/lattices/z/fccee_z_beam_1_positrons.json").fccee_p_ring
line_b2 = xt.load("acc-models-fcc-ee/lattices/z/fccee_z_beam_2_electrons.json").fccee_e_ring

# %%
# %%
import numpy as np
import matplotlib.pyplot as plt

half_crossing_angle = 15e-3  # rad

survey_b1 = line.survey(
    element0="ipa",
    theta0=+half_crossing_angle,
)

survey_b2 = line_b2.survey(
    element0="ipa",
    theta0=-half_crossing_angle,
)

# Coordinates relative to IPA
idx_b1 = list(survey_b1.name).index("ipa")
idx_b2 = list(survey_b2.name).index("ipa")

b1_X = survey_b1.X - survey_b1.X[idx_b1]
b1_Z = survey_b1.Z - survey_b1.Z[idx_b1]

b2_X_local = survey_b2.X - survey_b2.X[idx_b2]
b2_Z_local = survey_b2.Z - survey_b2.Z[idx_b2]

# Rotate the counter-rotating beam by 180 degrees around Y
b2_X = -b2_X_local
b2_Z = -b2_Z_local

fig, ax = plt.subplots()

ax.plot(b1_Z, b1_X, color="blue", label="Beam 1")
ax.plot(b2_Z, b2_X, color="red", label="Beam 2")

ax.scatter(0, 0, color="black", s=30, zorder=5)
ax.annotate("IPA", (0, 0), xytext=(8, 8), textcoords="offset points")

ax.set_xlabel("Z [m]")
ax.set_ylabel("X [m]")
ax.set_title("Beam reference trajectories around IPA")

ax.set_xlim(-1800, 1800)
ax.set_ylim(-40, 20)
ax.grid(alpha=0.3)
ax.legend()

plt.show()

# %%



line1 = line
line2 = line_b2

theta_init = 15.e-3
sv1 = line1.survey(theta0=theta_init)
sv2 = line2.survey(theta0=np.pi-theta_init)

ips = ['ipd','ipg','ipj','_end_point']
print("X err, \tY err, \tMET err")
for thing in ips:
    print(sv1['X',thing]-sv2['X',thing],sv1['Z',thing]-sv2['Z',thing],np.sqrt((sv1['X',thing]-sv2['X',thing])**2.+(sv1['Z',thing]-sv2['Z',thing])**2.))

import datetime

sv=sv1
sv_include_cols = [
                'name', 's', 'element_type', 'length',
                'X', 'Y', 'Z', 'theta', 'phi', 'psi',
                'isthick'
            ]
lname="POSITRONS"
sv_out = xt.Table({kk: sv[kk] for kk in sv_include_cols})
scalars_sv = dict(
    table='SURVEY',
    version='v107',
    line=lname,
    line_length=sv1['s'][-1],
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
for kk, vv in scalars_sv.items():
    sv_out[kk] = vv



sv=sv2
sv_include_cols = [
                'name', 's', 'element_type', 'length',
                'X', 'Y', 'Z', 'theta', 'phi', 'psi',
                'isthick'
            ]
lname="ELECTRONS"
sv_out = xt.Table({kk: sv[kk] for kk in sv_include_cols})
scalars_sv = dict(
    table='SURVEY',
    version='v107',
    line=lname,
    line_length=sv2['s'][-1],
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
for kk, vv in scalars_sv.items():
    sv_out[kk] = vv




import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def distance_to_curve(x1, y1, s1, x2, y2, s2):
    """
    For each point of curve 1, compute the minimum Euclidean
    distance to any point on curve 2.

    Parameters
    ----------
    x1, y1 : array-like
        Coordinates of curve 1.
    s1 : array-like
        Arc length coordinate of curve 1.
    x2, y2 : array-like
        Coordinates of curve 2.

    Returns
    -------
    s1 : ndarray
        Arc length positions along curve 1.
    dmin : ndarray
        Minimum distance from each point of curve 1 to curve 2.
    """

    curve2 = np.column_stack((x2, y2))
    tree = cKDTree(curve2)

    curve1 = np.column_stack((x1, y1))
    dmin, _ = tree.query(curve1)

    return np.asarray(s1), dmin


def plot_distance_vs_s(x1, y1, s1, x2, y2, s2,lab,logs=False,lll=None,absv=True):
    x2 = np.array(x2)
    y2 = np.array(y2)
    s2=np.array(s2)
    # Interpolating function
    df = pd.DataFrame({'x2': x2, 'y2': y2,'s2': s2})

    # keep first
    df = df.drop_duplicates(subset='s2', keep='first')
    s2 = df['s2'].to_numpy()
    x2 = df['x2'].to_numpy()
    y2 = df['y2'].to_numpy()

    fx = interp1d(s2, x2, kind='linear')
    fy = interp1d(s2, y2, kind='linear')

    # New x values
    s_new = np.linspace(s2.min(), s2.max(), 90000*4)

    # Interpolated y values
    x2 = fx(s_new)
    y2 = fy(s_new)
    if absv:
        s, d = distance_to_curve(x1, y1, s1, x2, y2,s_new)
    else:
        s = s1
        d= y1


    plt.figure(figsize=(8, 4))
    plt.plot(s, d)
    plt.ylabel(lab)
    if lll!=None:
        plt.ylim(lll)
    plt.xlabel("s coordinate in LCC (m)")
    plt.grid(True)
    if logs==True:
        plt.yscale('log')
    plt.tight_layout()
    return s, d
s_curve_vertical, d_curve_vertical = plot_distance_vs_s(
    np.array(sv1['X']) * 0,
    sv1['Y'],
    sv1['s'],
    np.array(sv2['X']) * 0,
    sv2['Y'],
    sv2['s'],
    "Abs. vertical distance LCC Beam 1 & 2 (m)",
)


plot_distance_vs_s(np.array(sv1['X'])*0, sv1['Y'], sv1['s'], np.array(sv2['X'])*0., np.array(sv2['Y'])*0., sv2['s'],"vertical distance LCC Beam 1 (m)",absv=False)


plot_distance_vs_s(np.array(sv2['X'])*0, sv2['Y'], sv2['s'], np.array(sv1['X'])*0., np.array(sv1['Y'])*0., sv1['s'],"vertical distance LCC Beam 2 (m)",absv=False)





s_curve_horizontal, d_curve_horizontal = plot_distance_vs_s(
    sv1['X'],
    sv1['Z'],
    sv1['s'],
    sv2['X'],
    sv2['Z'],
    sv2['s'],
    "Abs. horizontal distance LCC Beam 1 & 2 (m)",
)


plot_distance_vs_s(sv1['X'], sv1['Z'], sv1['s'], sv2['X'], sv2['Z'], sv2['s'],"Abs. distance LCC Beam 1 & 2 (m)",lll=[0.35-2e-3,0.35+5e-3])


# %%
# Compare the separations stored in Elements_Flat with the distances obtained
# from the original nearest-point-on-trajectory calculation above.
flat_distance_comparison = pd.read_excel(
    workbook_path,
    sheet_name="Elements_Flat",
    header=1,
)
flat_distance_comparison.columns = [
    str(column).replace("\n", " ").strip()
    for column in flat_distance_comparison.columns
]

comparison_columns = [
    "Old Name",
    "B1 s [m]",
    "Length [m]",
    "B2 element name",
    "Centre horizontal separation [m]",
    "Centre vertical separation [m]",
]
missing_comparison_columns = [
    column for column in comparison_columns
    if column not in flat_distance_comparison.columns
]
if missing_comparison_columns:
    raise KeyError(
        "Cannot compare Excel and trajectory distances. Missing columns: "
        f"{missing_comparison_columns}"
    )

for column in [
    "B1 s [m]",
    "Length [m]",
    "Centre horizontal separation [m]",
    "Centre vertical separation [m]",
]:
    flat_distance_comparison[column] = pd.to_numeric(
        flat_distance_comparison[column], errors="coerce"
    )

flat_distance_comparison = flat_distance_comparison.dropna(
    subset=[
        "B2 element name",
        "B1 s [m]",
        "Length [m]",
        "Centre horizontal separation [m]",
        "Centre vertical separation [m]",
    ]
).copy()
flat_distance_comparison["B1 centre s [m]"] = (
    flat_distance_comparison["B1 s [m]"]
    + 0.5 * flat_distance_comparison["Length [m]"]
)
flat_distance_comparison = flat_distance_comparison.sort_values(
    "B1 centre s [m]"
)


def interpolate_curve_distance(s_curve, distance_curve, s_at_magnets):
    curve = pd.DataFrame({
        "s": np.asarray(s_curve, dtype=float),
        "distance": np.asarray(distance_curve, dtype=float),
    })
    curve = (
        curve.dropna()
        .drop_duplicates(subset="s", keep="first")
        .sort_values("s")
    )
    return np.interp(
        np.asarray(s_at_magnets, dtype=float),
        curve["s"],
        curve["distance"],
    )


flat_distance_comparison["Trajectory horizontal separation [m]"] = (
    interpolate_curve_distance(
        s_curve_horizontal,
        d_curve_horizontal,
        flat_distance_comparison["B1 centre s [m]"],
    )
)
flat_distance_comparison["Trajectory vertical separation [m]"] = (
    interpolate_curve_distance(
        s_curve_vertical,
        d_curve_vertical,
        flat_distance_comparison["B1 centre s [m]"],
    )
)
flat_distance_comparison["Horizontal difference [m]"] = (
    flat_distance_comparison["Centre horizontal separation [m]"]
    - flat_distance_comparison["Trajectory horizontal separation [m]"]
)
flat_distance_comparison["Vertical difference [m]"] = (
    flat_distance_comparison["Centre vertical separation [m]"]
    - flat_distance_comparison["Trajectory vertical separation [m]"]
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 9),
    sharex=True,
    constrained_layout=True,
)

axes[0].plot(
    s_curve_horizontal,
    d_curve_horizontal,
    color="0.25",
    lw=1.0,
    label="Nearest B2 trajectory (original method)",
)
axes[0].scatter(
    flat_distance_comparison["B1 centre s [m]"],
    flat_distance_comparison["Centre horizontal separation [m]"],
    s=8,
    color="tab:blue",
    alpha=0.65,
    label="Elements_Flat: matched magnet centres",
)
axes[0].set_ylabel("Horizontal separation [m]")
axes[0].set_title("Horizontal inter-beam separation: catalogue vs trajectory")

axes[1].plot(
    s_curve_vertical,
    d_curve_vertical,
    color="0.25",
    lw=1.0,
    label="Nearest B2 trajectory (original method)",
)
axes[1].scatter(
    flat_distance_comparison["B1 centre s [m]"],
    flat_distance_comparison["Centre vertical separation [m]"],
    s=8,
    color="tab:orange",
    alpha=0.65,
    label="Elements_Flat: matched magnet centres",
)
axes[1].set_xlabel("B1 s at element centre [m]")
axes[1].set_ylabel("Vertical separation [m]")
axes[1].set_title("Vertical inter-beam separation: catalogue vs trajectory")

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

comparison_plot = output_folder / f"catalogue_vs_trajectory_distance_{energy}.png"
fig.savefig(comparison_plot, dpi=180)
print(f"\nCatalogue/trajectory comparison plot saved to: {comparison_plot}")

comparison_csv = output_folder / f"catalogue_vs_trajectory_distance_{energy}.csv"
flat_distance_comparison.to_csv(comparison_csv, index=False)
print(f"Point-by-point comparison saved to: {comparison_csv}")
plt.show()

# %%
