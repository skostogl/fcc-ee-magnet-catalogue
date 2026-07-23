# FCC-ee Magnet and Circuit Catalogues

This project creates magnet, circuit, corrector, BPM, and higher-multipole
catalogues from the FCC-ee Xsuite lattice models.

The generator automatically discovers the available energy lattices under
`acc-models-fcc-ee/lattices/`. The current model provides `Z`, `W`, `H`, and
`T` energies and reports lattice version `LCC_107.0.0`.

## Project layout

```text
000_create_catalogues.py          Main multi-energy catalogue generator
1_plot.py                         Cross-energy plotting example
acc-models-fcc-ee/                FCC-ee lattice-model checkout
FCC_Magnet_Report_<energy>.xlsx   Generated Excel reports
catalogues/
  <energy>/
    <version>/                    Generated text catalogues
plots/                            Generated comparison plots
```

Excel reports intentionally remain beside `000_create_catalogues.py`. Text
outputs are separated by energy and lattice version, for example:

```text
catalogues/z/LCC_107.0.0/
```

## Requirements

- Python 3.10+
- `xtrack`
- `numpy`
- `pandas`
- `openpyxl`
- `matplotlib`

The `acc-models-fcc-ee` directory is an external lattice-model checkout and is
not included in this repository. Place or clone it at the path shown above.

## Generate catalogues

Generate all discovered energies:

```bash
python 000_create_catalogues.py
```

Generate one energy:

```bash
python 000_create_catalogues.py --energy z
```

Each Excel workbook includes summaries, region and circuit catalogues,
element-by-element magnet data, higher multipoles, and a combined
corrector/BPM cell-assignment sheet.

The corrector/BPM sheet starts with:

1. full-ring totals;
2. totals grouped by `INSn`, `DSn`, and `ARCn`;
3. the complete position-sorted element list.

BPMs and correctors inherit the cell number of the quadrupole named in their
lattice element identifier. Nearby sextupoles inherit the nearest quadrupole
cell using the configured same-girder distance.

## Plot all energies

After generating the workbooks:

```bash
python 1_plot.py
```

The example reads `Elements_Flat` from every available Excel report and writes:

```text
plots/dipole_length_and_field_all_energies.png
```

The upper panel shows dipole length versus longitudinal position. The lower
panel shows dipole field in millitesla.
