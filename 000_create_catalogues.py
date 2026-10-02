# %%
import xtrack as xt
import numpy as np
import pandas as pd
import datetime
import re
import argparse
import subprocess
import sys
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

#%%

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent
LATTICE_ROOT = PROJECT_DIR / 'acc-models-fcc-ee' / 'lattices'
VERSION_FILE = PROJECT_DIR / 'acc-models-fcc-ee' / 'VERSION'

available_energies = sorted(
    p.name for p in LATTICE_ROOT.iterdir()
    if p.is_dir() and (p / f'fccee_{p.name}.json').exists()
)


#%%

parser = argparse.ArgumentParser(
    description='Create FCC-ee magnet and circuit catalogues.'
)
parser.add_argument(
    '--energy',
    choices=available_energies,
    help='Generate one energy only. Without this option all energies are generated.',
)
#args = parser.parse_args()
args, unknown = parser.parse_known_args()


if args.energy is None:
    print(f'Generating all available energies: {", ".join(e.upper() for e in available_energies)}')
    for energy in available_energies:
        print(f'\n{"=" * 78}\nENERGY {energy.upper()}\n{"=" * 78}')
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--energy', energy],
            cwd=PROJECT_DIR,
            check=True,
        )
    print('\nAll energy catalogues completed.')
    sys.exit(0)

mode             = args.energy
reference_radius = 0.035
lattice_version  = VERSION_FILE.read_text().strip()
lattice_model    = lattice_version.rsplit('.', 2)[0]



text_output_dir = PROJECT_DIR / 'catalogues' / mode / lattice_version
text_output_dir.mkdir(parents=True, exist_ok=True)

catalogue_file      = text_output_dir / f'Magnet_catalogue_{mode}.txt'
dipole_circuit_file = text_output_dir / f'Circuit_catalogue_{mode}_dipoles.txt'
quad_circuit_file   = text_output_dir / f'Circuit_catalogue_{mode}_quads.txt'
sext_circuit_file   = text_output_dir / f'Circuit_catalogue_{mode}_sext.txt'
rename_file         = text_output_dir / f'FCC_arc_rename_catalogue_{mode}.txt'

# ══════════════════════════════════════════════════════════════════════════════
# LOAD LATTICE
# ══════════════════════════════════════════════════════════════════════════════

beam1_lattice_file = (
    LATTICE_ROOT / mode / f'fccee_{mode}_beam_1_positrons.json'
)
if beam1_lattice_file.exists():
    line = xt.load(str(beam1_lattice_file)).fccee_p_ring
    lattice_source_file = beam1_lattice_file
else:
    lattice_source_file = LATTICE_ROOT / mode / f'fccee_{mode}.json'
    line = xt.load(str(lattice_source_file)).fccee_p_ring
tab    = line.get_table()
twissz = line.twiss4d()
brho   = line.particle_ref.p0c[0] / line.particle_ref.q0 / 299792458
circumference = float(tab.s[-1])
print(
    f'Loaded lattice: {lattice_model}, source = {lattice_source_file.name}, '
    f'circumference = {circumference:.3f} m, brho = {brho:.3f} Tm'
)

#%%

# ── Build fast twiss lookup ────────────────────────────────────────────────────
twiss_index = {name: i for i, name in enumerate(twissz.name)}

def get_twiss(name, col):
    idx = twiss_index.get(name)
    if idx is None:
        return np.nan
    try:
        return float(twissz[col][idx])
    except Exception:
        return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# REGION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

arc_defs = {
    'A1': ('end_ds_start_arc_ipa', 'end_arc_start_ds_ipb', 1, 2, 'ipa', 'ipb'),
    'A2': ('end_ds_start_arc_ipb', 'end_arc_start_ds_ipd', 2, 3, 'ipb', 'ipd'),
    'A3': ('end_ds_start_arc_ipd', 'end_arc_start_ds_ipf', 3, 4, 'ipd', 'ipf'),
    'A4': ('end_ds_start_arc_ipf', 'end_arc_start_ds_ipg', 4, 5, 'ipf', 'ipg'),
    'A5': ('end_ds_start_arc_ipg', 'end_arc_start_ds_iph', 5, 6, 'ipg', 'iph'),
    'A6': ('end_ds_start_arc_iph', 'end_arc_start_ds_ipj', 6, 7, 'iph', 'ipj'),
    'A7': ('end_ds_start_arc_ipj', 'end_arc_start_ds_ipl', 7, 8, 'ipj', 'ipl'),
    'A8': ('end_ds_start_arc_ipl', 'end_arc_start_ds_ipa', 8, 1, 'ipl', 'ipa'),
}

ip_sector = {'ipa': 1, 'ipb': 2, 'ipd': 3, 'ipf': 4,
             'ipg': 5, 'iph': 6, 'ipj': 7, 'ipl': 8}

S3_DS_OFFSET  = 100
S3_ARC_OFFSET = 200
S3_SAME_GIRDER_MAX_DIST = 3.0  # [m], nearby devices inherit nearest quad cell

# A proposed beam-2 element is accepted only when its longitudinal position is
# close to the nearest point on the interpolated beam-2 reference trajectory.
# This is deliberately different from the physical inter-beam distance, which
# can legitimately be several metres in an insertion.
B2_ELEMENT_MATCH_MAX_DS = 0.25  # [m], tolerance outside the B2 magnet span
B2_SURVEY_INTERPOLATION_POINTS = 360_000

KEEP_TYPES    = {'Quadrupole', 'RBend', 'Sextupole', 'Multipole', 'Marker'}
SKIP_PREFIXES = ('hcor_', 'vcor_', 'bpm_')

# ══════════════════════════════════════════════════════════════════════════════
# NAMING RULES  (scheme 3)
# ══════════════════════════════════════════════════════════════════════════════

ARC_QUAD_MAP = {
    'qd1a': 'MQD',  'qf2a': 'MQFA', 'qf3a': 'MQFB',
    'qd0i': "MQD", 'qd2i': "MQD",
    "qf0i": "MQFA", "qf1i": "MQFA", "qf3i": "MQFA", 
    'qd0j': "MQD", 'qd2j': "MQD",
    "qf0j": "MQFA", "qf1j": "MQFA", "qf3j": "MQFA", 
    "qd2c": "MQD",
    "qf0c": "MQFA", "qf3c": "MQFA",


    'qf1a': 'MQEA','qf1b': 'MQEB','qf1c': 'MQEC','qf1d': 'MQED',
    'qd0a': 'MQEDA','qd0b': 'MQEDB','qd0c': 'MQEDC',
    'qy1a': 'MQYA', 'qy1b': 'MQYB', 'qy1c': 'MQYC', 'qy1d': 'MQYD',
    'qx0a': 'MQXA', 'qx0b': 'MQXB', 'qx0c': 'MQXC',
    'qy1':  'MQYA', 'qy2':  'MQYB', 'qy3':  'MQYC', 'qy4':  'MQYD',
    'qx0':  'MQXA', 'qx1':  'MQXB', 'qx2':  'MQXC',
    'qd0m': 'MQMD', 'qf0m': 'MQMF', 'qd10m':'MQMD', 'qf9m': 'MQMF',
    'qd8m': 'MQMD', 'qd6m': 'MQMD', 'qf5m': 'MQMF', 'qd4m': 'MQMD',
    'qf3m': 'MQMF', 'qd2m': 'MQMD', 'qf1m': 'MQMF',
    'qf2r': 'MQE2', 'qf2l': 'MQE2', 'qd3':  'MQE3', 'qd4':  'MQE4',
    'qf5':  'MQE5', 'qd6r': 'MQE6', 'qd6l': 'MQE6', 'qd7':  'MQE7',
    'qf8':  'MQE8', 'qd9':  'MQE9', 'qf10': 'MQE10','qd11': 'MQE11',
    'qf12': 'MQE12','qf13r':'MQE13','qf13l':'MQE13','qd14r':'MQE14',
    'qd14l':'MQE14','qf15r':'MQE15','qf15l':'MQE15','qd16': 'MQE16',
    'qf17': 'MQE17','qd18r':'MQE18','qd18l':'MQE18','qf19': 'MQE19',
    'qd20': 'MQE20',
}

ARC_SEXT_MAP = {
    'sf2a': 'MSF', 'sf1a': 'MSF', 'sf1b': 'MSF', 'sf2b': 'MSF', 'sd1a': 'MSD', 'sd1b': 'MSD', 'sd2a': 'MSD','sd2b': 'MSD',
    'sf1c': 'MSF', 'sf1d': 'MSF', 'sf1f': 'MSF', 'sf1i': 'MSF', 'sd1c': 'MSD', 'sd1d': 'MSD', 'sd1f': 'MSD','sd1i': 'MSD',
    'sf2c': 'MSF', 'sf2d': 'MSF', 'sf2f': 'MSF', 'sf2i': 'MSF', 'sd2c': 'MSD', 'sd2d': 'MSD', 'sd2f': 'MSD','sd2i': 'MSD',
    'oct0': 'MOE',  'ocx1': 'MOXC', 'ocx2': 'MOXB', 'oct2': 'MOXA',
    'ocy1': 'MOYC', 'ocy2': 'MOYB', 'oct1': 'MOYA',
    'decf': 'MDF',  'decd': 'MDD',  'dec1': 'MDE',
    'sfx1': 'MSXF','sfx2': 'MSXF',
    'sdy1': 'MSYD','sdy2': 'MSYD',
    'sdm1': 'MSED', 'sfm2': 'MSEF', 'sf3m': 'MSMF',
}

DIPOLE_PREFIXES = ('mbd',)

INS_DIPOLE_MAP = {
    'b5ra': 'MBE5A','b5rb': 'MBE5B','b5l':  'MBE5', 'b6r':  'MBE6',
    'b6l':  'MBE6', 'b7ra': 'MBE7A','b7rb': 'MBE7B','b7l':  'MBE7',
    'b0al': 'MB0A','b0bl': 'MB0B','b0cl': 'MB0C','b0dl': 'MB0D',
    'b1lb': 'MBE1', 'b3l':  'MBE3', 'b4la': 'MBE4A','b4lb': 'MBE4B',
    'b4lc': 'MBE4C','b1ra': 'MBE1A','b1rb': 'MBE1B','b1rc': 'MBE1C',
    'b1rd': 'MBE1D','b1re': 'MBE1E','b3r':  'MBE3', 'b4ra': 'MBE4A',
    'b4rb': 'MBE4B','df1a': 'MBD',  'ds1a': 'MBD',
    'vsei1': 'MBS', 'vsei2': 'MBS',
}

SC_PREFIXES = ('scrabl', 'scrabr', 'scabl', 'scabr')
SC_LETTERS  = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def get_sc_type(name):
    for prefix in SC_PREFIXES:
        if name.startswith(prefix):
            try:
                n = int(name.split('.')[1])
                return ('MSC', SC_LETTERS[n % 4])
            except Exception:
                return ('MSC', 'A')
    return None


def get_new_type(name, type_map):
    for prefix, new_type in sorted(type_map.items(), key=lambda x: len(x[0]), reverse=True):
        if name.startswith(prefix):
            return new_type
    return None


def get_dipole_letter(old_name):
    for prefix in DIPOLE_PREFIXES:
        if old_name.startswith(prefix):
            suffix = old_name[len(prefix):]
            if suffix and suffix[0] in 'abc':
                return suffix[0].upper()
            return 'A'
    return 'A'


def make_name(mtype, cell, side, label, dletter=None):
    if dletter:
        return f"{mtype}.{dletter}{cell}{side}{label}"
    return f"{mtype}.{cell}{side}{label}"


def scheme3_name(old_name, etype, cell, side, sector):
    label = str(sector)
    if etype == 'Quadrupole':
        t = get_new_type(old_name, ARC_QUAD_MAP) or 'NOT YET DEFINED'
        return make_name(t, cell, side, label)
    elif etype == 'RBend':
        ins_type = get_new_type(old_name, INS_DIPOLE_MAP)
        if ins_type:
            return make_name(ins_type, cell, side, label)
        if any(old_name.startswith(p) for p in DIPOLE_PREFIXES):
            return make_name('MB', cell, side, label, get_dipole_letter(old_name))
        return f"NOT YET DEFINED.{cell}{side}{label}"
    elif etype in ('Sextupole', 'Multipole'):
        sc = get_sc_type(old_name)
        if sc is not None:
            mtype, letter = sc
            return make_name(mtype, cell, side, label, dletter=letter)
        t = get_new_type(old_name, ARC_SEXT_MAP) or 'NOT YET DEFINED'
        return make_name(t, cell, side, label)
    return f"NOT YET DEFINED.{cell}{side}{label}"

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def circular_dist(s1, s2, C):
    d = abs(s1 - s2)
    return min(d, C - d)


def circular_midpoint(s_start, s_end, C):
    if s_end >= s_start:
        return 0.5 * (s_start + s_end)
    return ((s_start + s_end + C) * 0.5) % C


def shifted(s, s_ref, wrap):
    return s + circumference if (wrap and s < s_ref) else s


def get_s(name):
    return float(twissz.rows[name]['s'][0])


def get_elements_in_range(s_low, s_high, wrap=False):
    result = []
    for name, s, et in zip(tab.name, tab.s, tab.element_type):
        if et not in KEEP_TYPES or name.startswith(SKIP_PREFIXES):
            continue
        s = float(s)
        if wrap:
            if s >= s_low - 1e-6 or s <= s_high + 1e-6:
                result.append((name, s, et))
        else:
            if s_low - 1e-6 <= s <= s_high + 1e-6:
                result.append((name, s, et))
    return result


def get_arc_data(arc_label):
    start_mk, end_mk, *_ = arc_defs[arc_label]
    s0       = get_s(start_mk)
    s1       = get_s(end_mk)
    wrap     = s1 < s0
    elements = get_elements_in_range(s0, s1, wrap)
    sh       = lambda s: shifted(s, s0, wrap)
    elements = sorted(elements, key=lambda x: sh(x[1]))
    quads    = [(n, s) for n, s, et in elements if et == 'Quadrupole']
    return elements, quads, s0, s1, wrap


def find_central_quad(quads, s0, s1):
    s_mid = circular_midpoint(s0, s1, circumference)
    mqfbs = [(n, s) for n, s in quads if n.startswith('qf3a')]
    return min(mqfbs, key=lambda x: circular_dist(x[1], s_mid, circumference))

# ── Install midpoint markers ───────────────────────────────────────────────────
marker_pos = {name: float(s) for name, s, et in zip(tab.name, tab.s, tab.element_type)
              if str(et).lower() == 'marker'}

pat_start = re.compile(r'^end_ds_start_straight_(ip[a-z]+)$')
pat_end   = re.compile(r'^end_straight_start_ds_(ip[a-z]+)$')

straight_markers = {}
for name, s in marker_pos.items():
    m = pat_start.match(name)
    if m:
        straight_markers.setdefault(m.group(1), {})['start'] = (name, s)
    m = pat_end.match(name)
    if m:
        straight_markers.setdefault(m.group(1), {})['end'] = (name, s)

for ip in sorted(straight_markers):
    info = straight_markers[ip]
    if 'start' not in info or 'end' not in info:
        continue
    mk = f'mid_{ip}'
    if mk in line.element_names:
        continue
    s_mid = circular_midpoint(info['start'][1], info['end'][1], circumference)
    line.insert_element(name=mk, element=xt.Marker(), at_s=s_mid)

tab    = line.get_table()
twissz = line.twiss4d()
twiss_index = {name: i for i, name in enumerate(twissz.name)}

# ══════════════════════════════════════════════════════════════════════════════
# QUAD LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def upstream_quad_R(es, quads):
    candidates = [(n, s) for n, s in quads if s <= es + 1e-6]
    return max(candidates, key=lambda x: x[1]) if candidates else None


def upstream_quad_L(es, quads):
    candidates = [(n, s) for n, s in quads if s >= es - 1e-6]
    return min(candidates, key=lambda x: x[1]) if candidates else None


def nearest_quad(es, quads):
    if not quads:
        return None
    return min(quads, key=lambda x: circular_dist(es, x[1], circumference))

# ══════════════════════════════════════════════════════════════════════════════
# SCHEME 3 — naming per arc
# ══════════════════════════════════════════════════════════════════════════════

def scheme3(arc_label):
    _, _, rs, ls, right_ip, left_ip = arc_defs[arc_label]
    sector_r = ip_sector[right_ip]
    sector_l = ip_sector[left_ip]

    s_ip_r  = get_s(f'mid_{right_ip}')
    s_ip_l  = get_s(f'mid_{left_ip}')
    s_ds_r  = get_s(f'end_straight_start_ds_{right_ip}')
    s_arc_r = get_s(f'end_ds_start_arc_{right_ip}')
    s_arc_l = get_s(f'end_arc_start_ds_{left_ip}')
    s_ds_l  = get_s(f'end_ds_start_straight_{left_ip}')

    wrap_ins_r = s_ds_r  < s_ip_r
    wrap_ds_r  = s_arc_r < s_ds_r
    wrap_ins_l = s_ip_l  < s_ds_l
    wrap_ds_l  = s_ds_l  < s_arc_l

    elements_arc, quads_arc, s0, s1, wrap_arc = get_arc_data(arc_label)
    central_name, central_s = find_central_quad(quads_arc, s0, s1)

    ins_r = sorted(
        [(n, float(s), et) for n, s, et in
         get_elements_in_range(s_ip_r, s_ds_r, wrap=wrap_ins_r)
         if (not wrap_ins_r and float(s) > s_ip_r + 1e-6 and float(s) < s_ds_r - 1e-6)
         or (wrap_ins_r and (float(s) > s_ip_r + 1e-6 or float(s) < s_ds_r - 1e-6))],
        key=lambda x: x[1])
    quads_ins_r = [(n, s) for n, s, et in ins_r if et == 'Quadrupole']

    ds_r = sorted(
        [(n, float(s), et) for n, s, et in
         get_elements_in_range(s_ds_r, s_arc_r, wrap=wrap_ds_r)
         if (not wrap_ds_r and float(s) >= s_ds_r - 1e-6 and float(s) < s_arc_r - 1e-6)
         or (wrap_ds_r and (float(s) >= s_ds_r - 1e-6 or float(s) < s_arc_r - 1e-6))],
        key=lambda x: x[1])
    quads_ds_r = [(n, s) for n, s, et in ds_r if et == 'Quadrupole']

    arc_r = sorted(
        [(n, float(s), et) for n, s, et in elements_arc
         if float(s) >= s_arc_r - 1e-6 and float(s) <= central_s + 1e-6],
        key=lambda x: x[1])
    quads_arc_r = [(n, s) for n, s, et in arc_r if et == 'Quadrupole']

    qidx_r = {}
    for i, (n, s) in enumerate(quads_ins_r, start=1):
        qidx_r[n] = i
    for i, (n, s) in enumerate(quads_ds_r, start=0):
        qidx_r[n] = S3_DS_OFFSET + i
    for i, (n, s) in enumerate(quads_arc_r, start=0):
        qidx_r[n] = S3_ARC_OFFSET + i

    all_elems_r   = ins_r + ds_r + arc_r
    all_quads_r_s = quads_ins_r + quads_ds_r + quads_arc_r

    arc_l = sorted(
        [(n, float(s), et) for n, s, et in elements_arc
         if float(s) > central_s + 1e-6 and float(s) <= s_arc_l + 1e-6],
        key=lambda x: x[1])
    quads_arc_l = [(n, s) for n, s, et in arc_l if et == 'Quadrupole']

    ds_l = sorted(
        [(n, float(s), et) for n, s, et in
         get_elements_in_range(s_arc_l, s_ds_l, wrap=wrap_ds_l)
         if (not wrap_ds_l and float(s) > s_arc_l + 1e-6 and float(s) <= s_ds_l + 1e-6)
         or (wrap_ds_l and (float(s) > s_arc_l + 1e-6 or float(s) <= s_ds_l + 1e-6))],
        key=lambda x: x[1])
    quads_ds_l = [(n, s) for n, s, et in ds_l if et == 'Quadrupole']

    ins_l = sorted(
        [(n, float(s), et) for n, s, et in
         get_elements_in_range(s_ds_l, s_ip_l, wrap=wrap_ins_l)
         if (not wrap_ins_l and float(s) > s_ds_l + 1e-6 and float(s) < s_ip_l - 1e-6)
         or (wrap_ins_l and (float(s) > s_ds_l + 1e-6 or float(s) < s_ip_l - 1e-6))],
        key=lambda x: x[1])
    quads_ins_l = [(n, s) for n, s, et in ins_l if et == 'Quadrupole']

    qidx_l = {}
    for i, (n, s) in enumerate(reversed(quads_ins_l), start=1):
        qidx_l[n] = i
    for i, (n, s) in enumerate(reversed(quads_ds_l), start=0):
        qidx_l[n] = S3_DS_OFFSET + i
    for i, (n, s) in enumerate(reversed(quads_arc_l), start=0):
        qidx_l[n] = S3_ARC_OFFSET + i

    all_elems_l   = arc_l + ds_l + ins_l
    all_quads_l_s = quads_arc_l + quads_ds_l + quads_ins_l

    rename = {}

    for qname in qidx_r:
        rename[qname] = scheme3_name(qname, 'Quadrupole', qidx_r[qname], 'R', sector_r)
    for qname in qidx_l:
        rename[qname] = scheme3_name(qname, 'Quadrupole', qidx_l[qname], 'L', sector_l)

    for ename, es, etype in all_elems_r:
        if etype in ('Quadrupole', 'Marker'):
            continue
        uq = None
        if etype == 'Sextupole':
            candidate = nearest_quad(es, all_quads_r_s)
            if (candidate is not None and
                    circular_dist(es, candidate[1], circumference)
                    <= S3_SAME_GIRDER_MAX_DIST + 1e-6):
                uq = candidate
        if uq is None:
            uq = upstream_quad_R(es, all_quads_r_s)
        if uq is None:
            # element is before the first quad on right side — use the first quad
            uq = min(all_quads_r_s, key=lambda x: x[1]) if all_quads_r_s else None
        rename[ename] = scheme3_name(ename, etype, qidx_r[uq[0]], 'R', sector_r) if uq else 'NOT FOUND'

    for ename, es, etype in all_elems_l:
        if etype in ('Quadrupole', 'Marker'):
            continue
        uq = None
        if etype == 'Sextupole':
            candidate = nearest_quad(es, all_quads_l_s)
            if (candidate is not None and
                    circular_dist(es, candidate[1], circumference)
                    <= S3_SAME_GIRDER_MAX_DIST + 1e-6):
                uq = candidate
        if uq is None:
            uq = upstream_quad_L(es, all_quads_l_s)
        if uq is None:
            # element is before the first quad on left side — use the first quad
            uq = min(all_quads_l_s, key=lambda x: x[1]) if all_quads_l_s else None
        rename[ename] = scheme3_name(ename, etype, qidx_l[uq[0]], 'L', sector_l) if uq else 'NOT FOUND'
    return rename

# ══════════════════════════════════════════════════════════════════════════════
# RUN NAMING — build master rename map
# ══════════════════════════════════════════════════════════════════════════════

master_rm3 = {}
print('Processing scheme 3 naming...')
for arc_label in arc_defs:
    master_rm3.update(scheme3(arc_label))
    print(f'  {arc_label} done')

# ══════════════════════════════════════════════════════════════════════════════
# BUILD RAW MAGNET DATAFRAME  (script 2 style + new_name + betx/bety)
# ══════════════════════════════════════════════════════════════════════════════

magnet_types = ['Quadrupole', 'RBend', 'Sextupole']
magnet_mask  = np.isin(tab.element_type, magnet_types)

records = []
for name, s in zip(tab.name[magnet_mask], tab.s[magnet_mask]):
    el    = line[name]
    etype = tab.rows[name]['element_type'][0]

    length = getattr(el, 'length', 0.0) or 0.0
    k0     = (getattr(el, 'h',  0.0) or 0.0) if etype == 'RBend' else (getattr(el, 'k0', 0.0) or 0.0)
    k1     = getattr(el, 'k1', 0.0) or 0.0
    k2     = getattr(el, 'k2', 0.0) or 0.0

    records.append({
        'name':         name,
        'new_name':     master_rm3.get(name, 'NOT YET DEFINED'),
        's':            float(s),
        'element_type': etype,
        'length':       length,
        'angle':        k0 * length,
        'k0':           k0,
        'bfield':       k0 * brho,
        'k1':           k1,
        'qgrad':        k1 * brho,
        'qfield':       abs(k1 * brho * reference_radius),
        'k2':           k2,
        'sgrad':        k2 * brho / 2.0,
        'sfield':       abs(k2 * brho * 0.5 * reference_radius**2),
        'betx':         get_twiss(name, 'betx'),
        'bety':         get_twiss(name, 'bety'),
    })

df_raw = pd.DataFrame(records)
df_raw['family'] = df_raw['name'].str.replace(r'[_\.]?\d+$', '', regex=True).str.upper()

# ═════════════════════════════════════════════════════════════════════════
# BEAM-1 SURVEY FOR THE FLAT MAGNET LIST
# ═════════════════════════════════════════════════════════════════════════

B1_SURVEY_THETA0 = 15e-3  # [rad], same initial angle as the optics survey plots
B1_SURVEY_ORIGIN = 'ipa'  # survey is shifted so that this marker sits at (0, 0, 0)

def add_beam1_survey(df, beam1):
    """Add the B1 survey (entry of each magnet) with IPA as the origin."""
    sv = beam1.survey(theta0=B1_SURVEY_THETA0)
    sv_index = {str(n): i for i, n in enumerate(sv['name'])}
    origin = sv_index[B1_SURVEY_ORIGIN]
    for axis in ('X', 'Y', 'Z'):
        coord = np.asarray(sv[axis], dtype=float)
        coord = coord - coord[origin]
        df[f'b1_survey_{axis}'] = [
            coord[sv_index[n]] if n in sv_index else np.nan for n in df['name']
        ]

add_beam1_survey(df_raw, line)

# ═════════════════════════════════════════════════════════════════════════
# BEAM-2 GEOMETRY MATCH FOR THE FLAT MAGNET LIST
# ═══════════════════════════════════════════════════════════════════════

B2_MATCH_COLUMNS = [
    'closest_b2_element',
    'b2_match_status',
    'b2_element_s',
    'b2_matched_s',
    'b2_longitudinal_offset',
    'b2_longitudinal_displacement',
    'interbeam_horizontal',
    'interbeam_vertical',
    'interbeam_3d',
    'distance_to_b2_trajectory',
]


def add_beam2_geometry_matches(df, beam1, beam2):
    """Match each beam-1 magnet to a nearby same-type beam-2 magnet.

    The dense interpolation is used only to locate the closest station on the
    beam-2 trajectory.  The reported element name and inter-beam distances are
    calculated from an actual beam-2 magnet survey point.  The longitudinal
    offset and status reveal cases where no beam-2 magnet is present nearby.
    """
    theta_init = 15e-3
    sv1 = beam1.survey(theta0=theta_init)
    sv2 = beam2.survey(theta0=np.pi - theta_init)

    s2 = np.asarray(sv2['s'], dtype=float)
    s1 = np.asarray(sv1['s'], dtype=float)
    xyz1_survey = np.column_stack([
        np.asarray(sv1['X'], dtype=float),
        np.asarray(sv1['Y'], dtype=float),
        np.asarray(sv1['Z'], dtype=float),
    ])
    xyz2 = np.column_stack([
        np.asarray(sv2['X'], dtype=float),
        np.asarray(sv2['Y'], dtype=float),
        np.asarray(sv2['Z'], dtype=float),
    ])

    # interp1d requires unique, increasing longitudinal coordinates.
    unique_s1, unique_idx1 = np.unique(s1, return_index=True)
    xyz1_unique = xyz1_survey[unique_idx1]
    unique_s2, unique_idx = np.unique(s2, return_index=True)
    xyz2_unique = xyz2[unique_idx]
    s2_dense = np.linspace(
        unique_s2[0], unique_s2[-1], B2_SURVEY_INTERPOLATION_POINTS
    )
    xyz2_dense = np.column_stack([
        interp1d(unique_s2, xyz2_unique[:, axis], kind='linear')(s2_dense)
        for axis in range(3)
    ])
    trajectory_tree = cKDTree(xyz2_dense)

    sv1_index = {str(name): i for i, name in enumerate(sv1['name'])}
    sv2_index = {str(name): i for i, name in enumerate(sv2['name'])}

    tab2 = beam2.get_table()
    candidates_by_type = {}
    for name, s, element_type in zip(tab2.name, tab2.s, tab2.element_type):
        if element_type not in magnet_types or str(name) not in sv2_index:
            continue
        candidates_by_type.setdefault(str(element_type), []).append({
            'name': str(name),
            's': float(s),
            'length': float(getattr(beam2[str(name)], 'length', 0.0) or 0.0),
        })

    result = {column: [] for column in B2_MATCH_COLUMNS}
    circumference_b2 = float(unique_s2[-1])
    used_b2_elements = set()

    for _, magnet in df.iterrows():
        name1 = str(magnet['name'])
        element_type = str(magnet['element_type'])
        idx1 = sv1_index.get(name1)

        if idx1 is None:
            result['closest_b2_element'].append('')
            result['b2_match_status'].append('B1 SURVEY POINT NOT FOUND')
            for column in B2_MATCH_COLUMNS[2:]:
                result[column].append(np.nan)
            continue

        b1_centre_s = (
            float(magnet['s']) + 0.5 * float(magnet['length'])
        ) % float(unique_s1[-1])
        xyz1 = np.array([
            np.interp(b1_centre_s, unique_s1, xyz1_unique[:, axis])
            for axis in range(3)
        ])

        distance_curve, dense_idx = trajectory_tree.query(xyz1)
        closest_s2 = float(s2_dense[dense_idx])

        candidates = candidates_by_type.get(element_type, [])
        if not candidates:
            result['closest_b2_element'].append('')
            result['b2_match_status'].append('NO B2 MAGNET OF SAME TYPE')
            result['b2_element_s'].append(np.nan)
            result['b2_matched_s'].append(np.nan)
            result['b2_longitudinal_offset'].append(np.nan)
            result['b2_longitudinal_displacement'].append(np.nan)
            result['interbeam_horizontal'].append(np.nan)
            result['interbeam_vertical'].append(np.nan)
            result['interbeam_3d'].append(np.nan)
            result['distance_to_b2_trajectory'].append(float(distance_curve))
            continue

        def distance_to_magnet_span(candidate):
            """Circular distance from closest_s2 to [start, start + length]."""
            best_offset = np.inf
            best_s = candidate['s']
            best_displacement = np.nan
            for shift in (-circumference_b2, 0.0, circumference_b2):
                start = candidate['s'] + shift
                end = start + candidate['length']
                matched_s = min(max(closest_s2, start), end)
                displacement = closest_s2 - matched_s
                offset = abs(displacement)
                if offset < best_offset:
                    best_offset = offset
                    best_s = matched_s % circumference_b2
                    best_displacement = displacement
            return float(best_offset), float(best_s), float(best_displacement)

        candidate_matches = [
            (candidate, *distance_to_magnet_span(candidate))
            for candidate in candidates
        ]
        # Enforce one-to-one correspondence. Without this, two neighbouring B1
        # magnets can select the same B2 magnet when their closest trajectory
        # stations fall inside the same long magnet span.
        candidate_matches.sort(key=lambda item: item[1])
        unused_matches = [
            item for item in candidate_matches
            if item[0]['name'] not in used_b2_elements
        ]

        if unused_matches:
            match, match_offset, matched_s2, match_displacement = unused_matches[0]
        else:
            match, match_offset, matched_s2, match_displacement = candidate_matches[0]

        if match_offset > B2_ELEMENT_MATCH_MAX_DS:
            result['closest_b2_element'].append('')
            result['b2_match_status'].append(
                f'NO B2 MATCH WITHIN {B2_ELEMENT_MATCH_MAX_DS:g} m'
            )
            result['b2_element_s'].append(match['s'])
            result['b2_matched_s'].append(matched_s2)
            result['b2_longitudinal_offset'].append(match_offset)
            result['b2_longitudinal_displacement'].append(match_displacement)
            result['interbeam_horizontal'].append(np.nan)
            result['interbeam_vertical'].append(np.nan)
            result['interbeam_3d'].append(np.nan)
            result['distance_to_b2_trajectory'].append(float(distance_curve))
            continue

        # After identifying the corresponding B2 magnet, compare the centres
        # of the two magnets. Matching still uses the complete B2 magnet span.
        b2_centre_s = (
            match['s'] + 0.5 * match['length']
        ) % circumference_b2
        xyz2_match = np.array([
            np.interp(b2_centre_s, unique_s2, xyz2_unique[:, axis])
            for axis in range(3)
        ])
        delta = xyz2_match - xyz1

        result['closest_b2_element'].append(match['name'])
        result['b2_match_status'].append('OK')
        used_b2_elements.add(match['name'])
        result['b2_element_s'].append(match['s'])
        result['b2_matched_s'].append(matched_s2)
        result['b2_longitudinal_offset'].append(match_offset)
        result['b2_longitudinal_displacement'].append(match_displacement)
        result['interbeam_horizontal'].append(float(np.hypot(delta[0], delta[2])))
        result['interbeam_vertical'].append(float(abs(delta[1])))
        result['interbeam_3d'].append(float(np.linalg.norm(delta)))
        result['distance_to_b2_trajectory'].append(float(distance_curve))

    for column in B2_MATCH_COLUMNS:
        df[column] = result[column]


beam1_geometry_file = LATTICE_ROOT / mode / f'fccee_{mode}_beam_1_positrons.json'
beam2_geometry_file = LATTICE_ROOT / mode / f'fccee_{mode}_beam_2_electrons.json'

if beam1_geometry_file.exists() and beam2_geometry_file.exists():
    print('Matching beam-1 magnets to the interpolated beam-2 survey...')
    line_b1_geometry = line
    line_b2_geometry = xt.load(str(beam2_geometry_file)).fccee_e_ring
    add_beam2_geometry_matches(df_raw, line_b1_geometry, line_b2_geometry)
    print(df_raw['b2_match_status'].value_counts(dropna=False).to_string())

    accepted_b2 = df_raw.loc[
        df_raw['b2_match_status'].eq('OK'), 'closest_b2_element'
    ]
    duplicated_b2 = accepted_b2[accepted_b2.duplicated(keep=False)]
    if not duplicated_b2.empty:
        duplicate_names = sorted(duplicated_b2.unique())
        raise RuntimeError(
            'Duplicate accepted B2 assignments found; workbook not written: '
            + ', '.join(duplicate_names[:20])
        )
else:
    print(f'Beam-2 geometry files are not available for energy {mode.upper()}.')
    for column in B2_MATCH_COLUMNS:
        df_raw[column] = np.nan
    df_raw['closest_b2_element'] = ''
    df_raw['b2_match_status'] = 'B2 LATTICE NOT AVAILABLE'

# ══════════════════════════════════════════════════════════════════════════════
# COMMENT FUNCTION  (unchanged from script 2)
# ══════════════════════════════════════════════════════════════════════════════

def get_comment(family, element_type):
    f = family.upper()
    if element_type == 'RBend':
        if f.startswith('MBD'):             return 'Main dipole'
        if f.startswith('MBD0') and 'RF' in f: return 'RF insertion - adjusted angle dipole'
        if f.startswith(('MBD8R','MBD9R')):   return 'RF insertion - right side dipole'
        if f.startswith('DF1'):             return 'Experimental insertion - dispersion suppressor dipole'
        if f.startswith('DS1') and 'RF' not in f: return 'Technical insertion (collimation/diagnostics/injection) - dispersion suppressor dipole'
        if f.startswith('DS') and 'RF' in f:    return 'RF insertion - dispersion suppressor dipole'
        if f.startswith('DOG') and 'RF' in f:   return 'RF insertion - dogleg dipole'
        if f.startswith('DOG') and 'COLL' in f: return 'Collimation insertion - dogleg dipole'
        if f.startswith('DOG') and 'DIAG' in f: return 'Diagnostics insertion - dogleg dipole'
        if f.startswith('VSEI'):            return 'Technical insertion - vertical separation dipole'
        if f.startswith(('DI0','DI1','DI2')): return 'Injection insertion - injection dipole'
        if f in {'B5L','B5RA','B5RB','B6L','B6R','B7L','B7RA','B7RB'}: return 'Experimental insertions - dipole'
        if f in {'B0AL','B0BL','B0CL','B0DL','B1LB','B3L','B4LA','B4LB','B4LC',
                 'B1RA','B1RB','B1RC','B1RD','B1RE','B3R','B4RA','B4RB'}: return 'Experimental insertions - dipole'
    elif element_type == 'Quadrupole':
        final_doublet = {'QF1AL','QF1AR','QF1BL','QF1BR','QF1CL','QF1CR','QF1DL','QF1DR',
                         'QD0AL','QD0AR','QD0BL','QD0BR','QD0CL','QD0CR'}
        ccs_quads = ('QF1','QD2','QF3','QD4','QF5','QD6','QF7','QD8','QF9','QD10',
                     'QF11','QD12','QF13','QD14','QF15','QD16','QF17','QD18','QD20',
                     'QF19','QD11','QD3','QD7','QD9','QF2','QF8','QF10','QF12',
                     'QF13','QF15','QF17','QF19')
        if f in {'QD1A','QD1AM','QF2A','QF3A'}:          return 'Main quadrupole'
        if f.startswith(('QF0M','QD0M','QF1M','QD2M','QF3M','QD4M','QF5M','QD6M','QD8M','QF9M','QD10M')): return 'Experimental insertion - matching quadrupole'
        if f in final_doublet:                            return 'Experimental insertion - final doublet quadrupole'
        if f.startswith(('QY','QX')):                     return 'Experimental insertion - final focus (outer) quadrupole'
        if any(f.startswith(q) for q in ccs_quads):
            if f.endswith(('F','FA','FB')):               return 'RF insertion - matching quadrupole'
            if f[-1] in ('C','D'):                        return 'Collimation/diagnostics/injection insertion - matching quadrupole'
            if f[-1] in ('I','J'):                        return 'Injection insertion - matching quadrupole'
            if f[-1] in ('L','R') and not f.startswith(('QY','QX')): return 'Experimental insertion - CCS quadrupole'
        if any(f.startswith(p) for p in ('QF0','QD0')) and f[-1] in ('C','D','F','I','J'): return 'Technical insertions - dedicated quadrupole in the transition cell between arc and DS'
    elif element_type == 'Sextupole':
        if f in {'SF1A','SF2A','SD1A','SD2A'}:            return 'Main sextupole'
        if f.startswith(('SF3M','SD3M')):                 return 'Experimental insertion - sextupole in matching section'
        if f.startswith('SCRAB'):                         return 'Experimental insertion - crab sextupole'
        if f.startswith(('SFX','SDY','SDM','SFM')):       return 'Experimental insertion - sextupole near IP'
        if any(f.startswith(p) for p in ('SF1','SF2','SD1','SD2')) and f.endswith(('FL','FR','BL','BR','CL','CR','DL','DR','IL','IR')): return 'Technical/Experimental insertions - dedicated sextupole in the transition cell between arc and DS'
        if f.startswith(('SF3','SD3','SF4')) and f[-1] in ('L','R'):
            if 'F' in f[-3:]:    return 'RF insertion - sextupole'
            if any(c in f for c in ('C','D')): return 'Coll/diag insertion - sextupole'
            if any(c in f for c in ('I','J')): return 'Injection insertion - sextupole'
    return ''

# ══════════════════════════════════════════════════════════════════════════════
# REGION TRANSITIONS  (used for both split catalogue and circuit building)
# ══════════════════════════════════════════════════════════════════════════════

sorted_markers = sorted(
    [(name, float(twissz.rows[name]['s'][0])) for name in twissz.name
     if any(k in name for k in ['end_straight_start_ds', 'end_ds_start_arc',
                                 'end_arc_start_ds',      'end_ds_start_straight'])],
    key=lambda x: x[1])
names_sorted_m = [n for n, _ in sorted_markers]

transitions_circuit = []
for i, (name, s) in enumerate(sorted_markers):
    ip = name.split('_')[-1]
    if 'end_straight_start_ds' in name:
        transitions_circuit.append((s, f'DS_{ip}_r'))
    elif 'end_ds_start_arc' in name:
        next_arc_end = next(n for n in names_sorted_m[i+1:] if 'end_arc_start_ds' in n)
        ip2 = next_arc_end.split('_')[-1]
        transitions_circuit.append((s, f'ARC_{ip}_{ip2}'))
    elif 'end_arc_start_ds' in name:
        transitions_circuit.append((s, f'DS_{ip}_l'))
    elif 'end_ds_start_straight' in name:
        transitions_circuit.append((s, f'INS_{ip}'))

transitions_display = []
for i, (name, s) in enumerate(sorted_markers):
    ip = name.split('_')[-1]
    n  = ip_sector.get(ip, '?')
    if 'end_straight_start_ds' in name:
        transitions_display.append((s, f'DS{n}R'))
    elif 'end_ds_start_arc' in name:
        next_arc_end = next(nn for nn in names_sorted_m[i+1:] if 'end_arc_start_ds' in nn)
        ip2 = next_arc_end.split('_')[-1]
        n2  = ip_sector.get(ip2, '?')
        transitions_display.append((s, f'ARC{n}{n2}'))
    elif 'end_arc_start_ds' in name:
        transitions_display.append((s, f'DS{n}L'))
    elif 'end_ds_start_straight' in name:
        transitions_display.append((s, f'INS{n}'))

arc_regions_ordered = []
seen_arcs = set()
for _, reg in transitions_circuit:
    if reg.startswith('ARC_') and reg not in seen_arcs:
        arc_regions_ordered.append(reg)
        seen_arcs.add(reg)

eps = 1e-9

def get_region_circuit(s_elem):
    # The first insertion crosses the lattice origin (s = 0).  Elements before
    # the first transition therefore belong to the region from the final
    # transition, not to the first region that starts later along the ring.
    region = transitions_circuit[-1][1]
    for s_trans, reg in transitions_circuit:
        if s_elem >= s_trans - eps:
            region = reg
        else:
            break
    return region

def get_region_display(s_elem):
    # Apply the same circular wrap-around rule to the user-facing region label.
    region = transitions_display[-1][1]
    for s_trans, reg in transitions_display:
        if s_elem >= s_trans - eps:
            region = reg
        else:
            break
    return region

df_raw['circuit_region'] = df_raw['s'].apply(get_region_circuit)
df_raw['region']         = df_raw['s'].apply(get_region_display)
df_raw['comment']        = df_raw.apply(lambda r: get_comment(r['family'], r['element_type']), axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTORS AND BPMS
# Cell attribution follows the explicitly associated quadrupole encoded in names
# such as hcor_qd1a.0 / vcor_qd1a.0 / bpm_qd1a.0.  This is also the nearest
# quadrupole within the same-girder distance for all but a few long RF girders.
# ══════════════════════════════════════════════════════════════════════════════

DEVICE_PREFIXES = {
    'bpm_':  ('BPM',                  'BPM'),
    'hcor_': ('Horizontal Corrector', 'MCH'),
    'vcor_': ('Vertical Corrector',   'MCV'),
}

def cell_parts(new_name):
    """Return cell number, side, and sector from a scheme-3 element name."""
    m = re.search(r'\.?[A-Z]?(\d+)([LR])(\d+)$', str(new_name))
    if not m:
        return np.nan, '', np.nan
    return int(m.group(1)), m.group(2), int(m.group(3))

def device_family(name, prefix):
    base = name[len(prefix):]
    base_family = re.sub(r'[_\.]?\d+$', '', base).upper()
    return f'{prefix[:-1].upper()}_{base_family}'

device_records = []
for name, s, etype in zip(tab.name, tab.s, tab.element_type):
    matched = next((p for p in DEVICE_PREFIXES if str(name).startswith(p)), None)
    if matched is None:
        continue

    device_type, new_prefix = DEVICE_PREFIXES[matched]
    assoc_quad = str(name)[len(matched):]
    assoc_quad_new = master_rm3.get(assoc_quad, 'NOT YET DEFINED')
    cell, side, sector = cell_parts(assoc_quad_new)
    suffix = (
        f'{int(cell)}{side}{int(sector)}'
        if pd.notna(cell) and side and pd.notna(sector)
        else 'NOT YET DEFINED'
    )
    new_name = f'{new_prefix}.{suffix}'

    el = line[name]
    length = float(getattr(el, 'length', 0.0) or 0.0)
    knl = np.asarray(getattr(el, 'knl', []), dtype=float)
    ksl = np.asarray(getattr(el, 'ksl', []), dtype=float)
    k0l = float(knl[0]) if len(knl) else 0.0
    k0sl = float(ksl[0]) if len(ksl) else 0.0

    try:
        quad_s = float(tab.rows[assoc_quad]['s'][0])
        quad_dist = circular_dist(float(s), quad_s, circumference)
    except Exception:
        quad_dist = np.nan

    if pd.notna(quad_dist) and quad_dist <= S3_SAME_GIRDER_MAX_DIST + 1e-6:
        attribution = 'Associated quadrupole — same girder'
    else:
        attribution = 'Associated quadrupole — name-linked long girder'

    device_records.append({
        'name':               str(name),
        'new_name':           new_name,
        'family':             device_family(str(name), matched),
        'device_type':        device_type,
        'region':             get_region_display(float(s)),
        's':                  float(s),
        'length':             length,
        'cell':               cell,
        'side':               side,
        'sector':             sector,
        'cell_attribution':   attribution,
        'associated_quad':    assoc_quad,
        'quad_distance':      quad_dist,
        'k0l':                k0l,
        'k0sl':               k0sl,
        'comment':            f'{device_type} associated with {assoc_quad}',
    })

df_devices = pd.DataFrame(device_records).sort_values(
    ['s', 'device_type', 'name']
).reset_index(drop=True)

total_bpms       = int((df_devices['device_type'] == 'BPM').sum())
total_hcorrectors = int((df_devices['device_type'] == 'Horizontal Corrector').sum())
total_vcorrectors = int((df_devices['device_type'] == 'Vertical Corrector').sum())

def compact_device_region(region):
    """Group left/right DS halves and label each arc by its starting sector."""
    m = re.fullmatch(r'DS(\d+)[LR]', str(region))
    if m:
        return f'DS{m.group(1)}'
    m = re.fullmatch(r'ARC(\d)(\d)', str(region))
    if m:
        return f'ARC{m.group(1)}'
    return str(region)

df_devices['summary_region'] = df_devices['region'].map(compact_device_region)

device_region_summary = (
    df_devices.groupby(['summary_region', 'device_type'])
    .size()
    .unstack(fill_value=0)
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FAMILY CATALOGUE  (summary table — one row per family)
# ══════════════════════════════════════════════════════════════════════════════

catalogue = df_raw.groupby('family', sort=False).agg(
    element_type = ('element_type', 'first'),
    count        = ('name',         'count'),
    length       = ('length',       'mean'),
    angle        = ('angle',        'mean'),
    k0           = ('k0',           'mean'),
    bfield       = ('bfield',       'mean'),
    k1           = ('k1',           'mean'),
    qgrad        = ('qgrad',        'mean'),
    qfield       = ('qfield',       'mean'),
    k2           = ('k2',           'mean'),
    sgrad        = ('sgrad',        'mean'),
    sfield       = ('sfield',       'mean'),
    betx_mean    = ('betx',         'mean'),
    bety_mean    = ('bety',         'mean'),
).reset_index()

catalogue = catalogue[~catalogue['family'].str.startswith(('VCOR','HCOR'))].reset_index(drop=True)

type_order = {'RBend': 0, 'Multipole': 1, 'Quadrupole': 2, 'Sextupole': 3}
catalogue = catalogue.sort_values(
    ['element_type', 'count'],
    key=lambda x: x.map(type_order) if x.name == 'element_type' else x,
    ascending=[True, False]
).reset_index(drop=True)

catalogue['comment'] = catalogue.apply(lambda r: get_comment(r['family'], r['element_type']), axis=1)

catalogue.columns = ['family','element_type','count','length [m]','angle [rad]','k0 [1/m]',
                     'bfield [T]','k1 [1/m2]','qgrad [T/m]','qfield [T]','k2 [1/m3]',
                     'sgrad [T/m2]','sfield [T]','betx_mean [m]','bety_mean [m]','comment']

total_dipoles     = catalogue.loc[catalogue['element_type'] == 'RBend',      'count'].sum()
total_quadrupoles = catalogue.loc[catalogue['element_type'] == 'Quadrupole', 'count'].sum()
total_sextupoles  = catalogue.loc[catalogue['element_type'] == 'Sextupole',  'count'].sum()

info = {
    'created':           str(datetime.datetime.now()),
    'machine':           'FCCee',
    'lattice model':     lattice_model,
    'type':              'twiss4d',
    'energy':            line.particle_ref.energy0[0],
    'circumference':     twissz.circumference,
    'qx':                twissz.qx,
    'qy':                twissz.qy,
    'dqx':               twissz.dqx,
    'dqy':               twissz.dqy,
    'brho':              brho,
    'reference_radius':  reference_radius,
    'total_dipoles':     int(total_dipoles),
    'total_quadrupoles': int(total_quadrupoles),
    'total_sextupoles':  int(total_sextupoles),
}

with open(catalogue_file, 'w') as f:
    f.write('# ── Machine parameters ─────────────────────────────────────────\n')
    for key, val in info.items():
        f.write(f'{key:<30}{val}\n')
    f.write('\n# ── Global family summary ───────────────────────────────────────\n\n')
    f.write(catalogue.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT CATALOGUE BY REGION  — family summary + element-level table
# ══════════════════════════════════════════════════════════════════════════════

split_family = df_raw.groupby(['family', 'region'], sort=False).agg(
    element_type  = ('element_type', 'first'),
    count         = ('name',         'count'),
    length        = ('length',       'mean'),
    angle         = ('angle',        'mean'),
    k0            = ('k0',           'mean'),
    bfield        = ('bfield',       'mean'),
    k1            = ('k1',           'mean'),
    qgrad         = ('qgrad',        'mean'),
    qfield        = ('qfield',       'mean'),
    k2            = ('k2',           'mean'),
    sgrad         = ('sgrad',        'mean'),
    sfield        = ('sfield',       'mean'),
    betx_mean     = ('betx',         'mean'),
    bety_mean     = ('bety',         'mean'),
    comment       = ('comment',      'first'),
).reset_index()

split_family.columns = ['family','region','element_type','count',
                         'length [m]','angle [rad]','k0 [1/m]','bfield [T]',
                         'k1 [1/m2]','qgrad [T/m]','qfield [T]',
                         'k2 [1/m3]','sgrad [T/m2]','sfield [T]',
                         'betx_mean [m]','bety_mean [m]','comment']

# element-level columns for the detail table
elem_cols_out = ['new_name','name','s','element_type','length',
                 'k0','bfield','k1','qgrad','k2','sgrad',
                 'betx','bety','comment']
elem_col_labels = ['new_name','old_name','s [m]','element_type','length [m]',
                   'k0 [1/m]','bfield [T]','k1 [1/m2]','qgrad [T/m]',
                   'k2 [1/m3]','sgrad [T/m2]','betx [m]','bety [m]','comment']

with open(catalogue_file, 'a') as f:
    f.write('\n\n')
    f.write('#' * 80 + '\n')
    f.write('### MAGNET CATALOGUE BY REGION\n')
    f.write('### Each region: (1) family summary, (2) element-level detail\n')
    f.write('#' * 80 + '\n')

    for region, grp_fam in split_family.groupby('region', sort=False):
        f.write(f'\n{"=" * 80}\n')
        f.write(f'### REGION: {region}\n')
        f.write(f'{"=" * 80}\n\n')

        # ── Family summary ────────────────────────────────────────────────────
        f.write('--- Family summary ---\n')
        grp_sorted = grp_fam.drop(columns='region').sort_values(
            ['element_type', 'count'],
            key=lambda x: x.map(type_order) if x.name == 'element_type' else x,
            ascending=[True, False])
        f.write(grp_sorted.to_string(index=False))
        f.write('\n\n')

        # ── Element-level detail ──────────────────────────────────────────────
        f.write('--- Element detail ---\n')
        grp_elem = df_raw[df_raw['region'] == region][elem_cols_out].copy()
        grp_elem = grp_elem.sort_values('s').reset_index(drop=True)
        grp_elem.columns = elem_col_labels
        f.write(grp_elem.to_string(index=False))
        f.write('\n')

print(f'Global + split catalogue saved to {catalogue_file}')

# ══════════════════════════════════════════════════════════════════════════════
# RENAME CATALOGUE  (flat list: old → new, with s and betx/bety)
# ══════════════════════════════════════════════════════════════════════════════

def get_cell_from_new(new_name):
    if not new_name or 'NOT' in new_name:
        return ''
    try:
        part = new_name.split('.')[1].lstrip('ABCDEFGH')
        cell = ''
        for c in part:
            if c.isdigit():
                cell += c
            else:
                break
        return cell
    except Exception:
        return ''

def marker_new_name(old_name):
    for ip, sector in ip_sector.items():
        if old_name in (ip, f'mid_{ip}') or old_name.startswith('ip.'):
            return f'ip{sector}'
    return old_name

def build_region_map_display():
    regions = []
    for arc_label, (start_mk, end_mk, *_) in arc_defs.items():
        s0 = get_s(start_mk);  s1 = get_s(end_mk)
        regions.append((s0, s1, arc_label, s1 < s0))
    for ip, n in ip_sector.items():
        s_ds_r0 = straight_markers[ip]['end'][1]
        s_ds_r1 = get_s(f'end_ds_start_arc_{ip}')
        s_ds_l0 = get_s(f'end_arc_start_ds_{ip}')
        s_ds_l1 = straight_markers[ip]['start'][1]
        s_ins0  = straight_markers[ip]['start'][1]
        s_ins1  = straight_markers[ip]['end'][1]
        regions += [
            (s_ds_r0, s_ds_r1, f'DS{n}R', s_ds_r1 < s_ds_r0),
            (s_ds_l0, s_ds_l1, f'DS{n}L', s_ds_l1 < s_ds_l0),
            (s_ins0,  s_ins1,  f'INS{n}', s_ins1  < s_ins0),
        ]
    return regions

region_map_display = build_region_map_display()

def get_region_label_fast(s, regions):
    for s0, s1, label, wrap in regions:
        if wrap:
            if s >= s0 - 1e-6 or s <= s1 + 1e-6:
                return label
        else:
            if s0 - 1e-6 <= s <= s1 + 1e-6:
                return label
    return 'UNKNOWN'

all_keep = [(name, float(s), et)
            for name, s, et in zip(tab.name, tab.s, tab.element_type)
            if et in KEEP_TYPES and not name.startswith(SKIP_PREFIXES)]

col_fixed = dict(region=10, old_name=28, s=12, etype=15, cell=8)
col_n     = 35
header    = (f"{'region':<{col_fixed['region']}} {'old_name':<{col_fixed['old_name']}} "
             f"{'s [m]':<{col_fixed['s']}} {'element_type':<{col_fixed['etype']}} "
             f"{'cell':<{col_fixed['cell']}} {'betx [m]':<12} {'bety [m]':<12} "
             f"{'scheme3_name':<{col_n}}")

lines = [header, '-' * len(header)]
for name, s, et in all_keep:
    region   = get_region_label_fast(s, region_map_display)
    betx_val = get_twiss(name, 'betx')
    bety_val = get_twiss(name, 'bety')
    betx_str = f'{betx_val:.2f}' if not np.isnan(betx_val) else 'N/A'
    bety_str = f'{bety_val:.2f}' if not np.isnan(bety_val) else 'N/A'
    if et == 'Marker':
        new_name = marker_new_name(name)
        cell     = ''
    else:
        new_name = master_rm3.get(name, 'NOT YET DEFINED')
        cell     = get_cell_from_new(new_name)
    lines.append(
        f"{region:<{col_fixed['region']}} {name:<{col_fixed['old_name']}} "
        f"{s:<{col_fixed['s']}.4f} {et:<{col_fixed['etype']}} "
        f"{cell:<{col_fixed['cell']}} {betx_str:<12} {bety_str:<12} "
        f"{new_name:<{col_n}}")

with open(rename_file, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Rename catalogue saved to {rename_file} ({len(all_keep)} elements)')

# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT CATALOGUES  — dipoles, quadrupoles, sextupoles
# All include new_name column.
# ══════════════════════════════════════════════════════════════════════════════

arc_dipole_families = {
    'MBD',
    'MBD0L_RF','MBD1L_RF','MBD2L_RF','MBD3L_RF',
    'MBD3R_RF','MBD0R_RF','DS1R_RF','DL9R_RF','DL8R_RF','DL0R_RF',
}

# ── Dipole circuits ────────────────────────────────────────────────────────────
circuit_rows = []
for arc in arc_regions_ordered:
    _, ip1, ip2 = arc.split('_')
    ds1 = f'DS_{ip1}_r';  ds2 = f'DS_{ip2}_l'
    circuit_name = f'CIRCUIT_{ip1}_{ip2}'
    mask = (
        (df_raw['element_type'] == 'RBend') &
        (df_raw['family'].isin(arc_dipole_families)) &
        (df_raw['circuit_region'].isin([ds1, arc, ds2]))
    )
    df_c = df_raw[mask].copy().sort_values('s').reset_index(drop=True)
    df_c['circuit'] = circuit_name;  df_c['arc'] = arc
    df_c['from_ds'] = ds1;           df_c['to_ds'] = ds2
    circuit_rows.append(df_c)

df_circuits = pd.concat(circuit_rows, ignore_index=True)
df_circuits_out = df_circuits[[
    'circuit','from_ds','arc','to_ds',
    'new_name','name','family','s','circuit_region',
    'length','angle','k0','bfield','betx','bety','comment'
]].copy()
df_circuits_out.columns = [
    'circuit','from_ds','arc','to_ds',
    'new_name','old_name','family','s [m]','region',
    'length [m]','angle [rad]','k0 [1/m]','bfield [T]',
    'betx [m]','bety [m]','comment'
]

with open(dipole_circuit_file, 'w') as f:
    for circuit_name, group in df_circuits_out.groupby('circuit', sort=False):
        row = group.iloc[0]
        f.write(f'\n{"=" * 80}\n')
        f.write(f'### CIRCUIT: {circuit_name}  [{row["from_ds"]} + {row["arc"]} + {row["to_ds"]}]\n')
        f.write(f'{"=" * 80}\n')
        f.write(f'{"total_dipoles":<30}{len(group)}\n\n')
        f.write(group.drop(columns=['circuit','from_ds','arc','to_ds']).to_string(index=False))
        f.write('\n')

# individually powered dipoles
all_arc_circuit_names = set(df_circuits_out['old_name'])
df_dip_remaining = df_raw[
    (df_raw['element_type'] == 'RBend') &
    (~df_raw['name'].isin(all_arc_circuit_names))
].copy().sort_values('s').reset_index(drop=True)

with open(dipole_circuit_file, 'a') as f:
    f.write(f'\n\n{"#" * 80}\n')
    f.write('### INDIVIDUALLY POWERED DIPOLES\n')
    f.write(f'{"#" * 80}\n')
    f.write(f'{"total_individually_powered":<30}{len(df_dip_remaining)}\n\n')
    f.write(df_dip_remaining[['new_name','name','family','s','region',
                               'length','angle','k0','bfield',
                               'betx','bety','comment']].to_string(index=False))
    f.write('\n')

print(f'Dipole circuit catalogue saved to {dipole_circuit_file}')

# ── Quadrupole circuits ────────────────────────────────────────────────────────
arc_quad_families_qf = {'QF2A','QF3A'}
arc_quad_families_qd = {'QD1A','QD1AM'}
arc_quad_families    = arc_quad_families_qf | arc_quad_families_qd

quad_circuit_rows = []
for arc in arc_regions_ordered:
    _, ip1, ip2 = arc.split('_')
    ds1 = f'DS_{ip1}_r';  ds2 = f'DS_{ip2}_l'
    arc_mask = df_raw['circuit_region'] == arc
    s_mid = (df_raw[arc_mask]['s'].min() + df_raw[arc_mask]['s'].max()) / 2
    for polarity, families in [('qf', arc_quad_families_qf), ('qd', arc_quad_families_qd)]:
        for side in ['l', 'r']:
            circuit_name = f'CIRCUIT_{ip1}_{ip2}_{side}_{polarity}'
            if side == 'l':
                mask = ((df_raw['element_type'] == 'Quadrupole') &
                        (df_raw['family'].isin(families)) &
                        ((df_raw['circuit_region'] == ds1) |
                         ((df_raw['circuit_region'] == arc) & (df_raw['s'] < s_mid))))
            else:
                mask = ((df_raw['element_type'] == 'Quadrupole') &
                        (df_raw['family'].isin(families)) &
                        (((df_raw['circuit_region'] == arc) & (df_raw['s'] >= s_mid)) |
                         (df_raw['circuit_region'] == ds2)))
            df_c = df_raw[mask].copy().sort_values('s').reset_index(drop=True)
            df_c['circuit'] = circuit_name;  df_c['arc'] = arc
            df_c['from_ds'] = ds1;           df_c['to_ds'] = ds2
            df_c['side'] = side;             df_c['polarity'] = polarity
            quad_circuit_rows.append(df_c)

df_quad_circuits = pd.concat(quad_circuit_rows).reset_index(drop=True)

with open(quad_circuit_file, 'w') as f:
    f.write(f'# FCC-ee Z Quadrupole Circuit Catalogue\n')
    f.write(f'# Lattice model: {lattice_model}\n')
    f.write(f'# Energy: {mode.upper()}, Brho = {brho:.6f} T·m\n')
    f.write(f'# Reference radius: {reference_radius*1000:.1f} mm\n\n')

for arc in arc_regions_ordered:
    _, ip1, ip2 = arc.split('_')
    for polarity, families in [('qf', arc_quad_families_qf), ('qd', arc_quad_families_qd)]:
        for side in ['l', 'r']:
            circuit_name = f'CIRCUIT_{ip1}_{ip2}_{side}_{polarity}'
            df_c = df_quad_circuits[df_quad_circuits['circuit'] == circuit_name]
            pol_str = 'focusing' if polarity == 'qf' else 'defocusing'
            side_str = 'left' if side == 'l' else 'right'
            with open(quad_circuit_file, 'a') as f:
                f.write(f'\n{"=" * 80}\n')
                f.write(f'### CIRCUIT: {circuit_name}  [{side_str} half of {arc} — {pol_str}]\n')
                f.write(f'{"=" * 80}\n')
                f.write(f'{"total_quads":<30}{len(df_c)}\n\n')
                f.write(df_c[['new_name','name','family','s','length',
                               'k1','qgrad','qfield','betx','bety',
                               'comment']].to_string(index=False))
                f.write('\n')

all_quad_circuit_names = set(df_quad_circuits['name'])
df_quad_remaining = df_raw[
    (df_raw['element_type'] == 'Quadrupole') &
    (~df_raw['name'].isin(all_quad_circuit_names))
].copy().sort_values('s').reset_index(drop=True)

with open(quad_circuit_file, 'a') as f:
    f.write(f'\n\n{"#" * 80}\n')
    f.write('### INDIVIDUALLY POWERED QUADRUPOLES\n')
    f.write(f'{"#" * 80}\n')
    f.write(f'{"total_individually_powered":<30}{len(df_quad_remaining)}\n\n')
    f.write(df_quad_remaining[['new_name','name','family','s','region',
                                'length','k1','qgrad','qfield',
                                'betx','bety','comment']].to_string(index=False))
    f.write('\n')

print(f'Quadrupole circuit catalogue saved to {quad_circuit_file}')

# ── Sextupole circuits ────────────────────────────────────────────────────────
arc_sext_families_map = {
    'SF1A': 'sf1a', 'SF2A': 'sf2a', 'SD1A': 'sd1a', 'SD2A': 'sd2a',
}
arc_sext_focusing   = {'SF1A', 'SF2A'}

sext_circuit_rows = []
for arc in arc_regions_ordered:
    _, ip1, ip2 = arc.split('_')
    ds1 = f'DS_{ip1}_r';  ds2 = f'DS_{ip2}_l'
    arc_mask = df_raw['circuit_region'] == arc
    s_mid = (df_raw[arc_mask]['s'].min() + df_raw[arc_mask]['s'].max()) / 2
    for family, fam_label in arc_sext_families_map.items():
        for side in ['l', 'r']:
            circuit_name = f'CIRCUIT_{ip1}_{ip2}_{side}_{fam_label}'
            if side == 'l':
                mask = ((df_raw['element_type'] == 'Sextupole') &
                        (df_raw['family'] == family) &
                        ((df_raw['circuit_region'] == ds1) |
                         ((df_raw['circuit_region'] == arc) & (df_raw['s'] < s_mid))))
            else:
                mask = ((df_raw['element_type'] == 'Sextupole') &
                        (df_raw['family'] == family) &
                        (((df_raw['circuit_region'] == arc) & (df_raw['s'] >= s_mid)) |
                         (df_raw['circuit_region'] == ds2)))
            df_c = df_raw[mask].copy().sort_values('s').reset_index(drop=True)
            df_c['circuit'] = circuit_name;  df_c['arc'] = arc;  df_c['side'] = side
            df_c['family_label'] = fam_label
            sext_circuit_rows.append(df_c)

df_sext_circuits = pd.concat(sext_circuit_rows).reset_index(drop=True)

with open(sext_circuit_file, 'w') as f:
    f.write(f'# FCC-ee Z Sextupole Circuit Catalogue\n')
    f.write(f'# Lattice model: {lattice_model}\n')
    f.write(f'# Energy: {mode.upper()}, Brho = {brho:.6f} T·m\n')
    f.write(f'# Reference radius: {reference_radius*1000:.1f} mm\n\n')

for arc in arc_regions_ordered:
    _, ip1, ip2 = arc.split('_')
    for family, fam_label in arc_sext_families_map.items():
        pol_str = 'focusing' if family in arc_sext_focusing else 'defocusing'
        for side in ['l', 'r']:
            circuit_name = f'CIRCUIT_{ip1}_{ip2}_{side}_{fam_label}'
            df_c = df_sext_circuits[df_sext_circuits['circuit'] == circuit_name]
            side_str = 'left' if side == 'l' else 'right'
            with open(sext_circuit_file, 'a') as f:
                f.write(f'\n{"=" * 80}\n')
                f.write(f'### CIRCUIT: {circuit_name}  [{side_str} half of {arc} — {pol_str} ({family})]\n')
                f.write(f'{"=" * 80}\n')
                f.write(f'{"total_sextupoles":<30}{len(df_c)}\n\n')
                f.write(df_c[['new_name','name','family','s','length',
                               'k2','sgrad','sfield','betx','bety',
                               'comment']].to_string(index=False))
                f.write('\n')

all_sext_circuit_names = set(df_sext_circuits['name'])
df_sext_remaining = df_raw[
    (df_raw['element_type'] == 'Sextupole') &
    (~df_raw['name'].isin(all_sext_circuit_names))
].copy().sort_values('s').reset_index(drop=True)

with open(sext_circuit_file, 'a') as f:
    f.write(f'\n\n{"#" * 80}\n')
    f.write('### INDIVIDUALLY POWERED SEXTUPOLES\n')
    f.write(f'{"#" * 80}\n')
    f.write(f'{"total_individually_powered":<30}{len(df_sext_remaining)}\n\n')
    f.write(df_sext_remaining[['new_name','name','family','s','region',
                                'length','k2','sgrad','sfield',
                                'betx','bety','comment']].to_string(index=False))
    f.write('\n')

print(f'Sextupole circuit catalogue saved to {sext_circuit_file}')
print('\nAll done.')
# %%
# ══════════════════════════════════════════════════════════════════════════════
# ENRICH CATALOGUE WITH NEW NAME TYPE, REGIONS, CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════

def get_new_name_type(new_name):
    """Strip cell/sector number from new_name, keep type prefix.
    e.g. MQD.108R1 → MQD,  MB.A108R1 → MB.A,  MSC.A38R1 → MSC.A"""
    if not new_name or 'NOT' in str(new_name):
        return str(new_name)
    try:
        parts = str(new_name).split('.')
        prefix = parts[0]
        if len(parts) < 2:
            return prefix
        letter = re.match(r'^([A-Z]*)', parts[1]).group(1)
        return f"{prefix}.{letter}" if letter else prefix
    except Exception:
        return str(new_name)

# family → new name type (no cell/sector)
fam_new_name = {}
for fam, grp in df_raw.groupby('family'):
    nn = grp['new_name'].dropna()
    nn = nn[~nn.str.contains('NOT', na=False)]
    fam_new_name[fam] = get_new_name_type(nn.iloc[0]) if len(nn) else 'NOT YET DEFINED'

# family → regions (compact)
fam_regions = {}
for fam, grp in df_raw.groupby('family'):
    regs     = sorted(grp['region'].unique())
    arc_regs = [r for r in regs if r.startswith('ARC')]
    ds_regs  = [r for r in regs if r.startswith('DS')]
    ins_regs = [r for r in regs if r.startswith('INS')]
    parts = []
    if len(arc_regs) == 8: parts.append('All ARCs')
    elif arc_regs:          parts.append(', '.join(arc_regs))
    if len(ds_regs) == 16:  parts.append('All DS')
    elif ds_regs:            parts.append(', '.join(ds_regs[:4]) + ('...' if len(ds_regs) > 4 else ''))
    if ins_regs:             parts.append(', '.join(ins_regs[:3]) + ('...' if len(ins_regs) > 3 else ''))
    fam_regions[fam] = ' | '.join(parts) if parts else ', '.join(regs[:3])

# family → circuits (compact summary)
fam_circuits = {}
for _, row in df_circuits_out.iterrows():
    fam_circuits.setdefault(row['family'], set()).add(row['circuit'])
for _, row in df_quad_circuits.iterrows():
    fam_circuits.setdefault(row['family'], set()).add(row['circuit'])
for _, row in df_sext_circuits.iterrows():
    fam_circuits.setdefault(row['family'], set()).add(row['circuit'])

fam_circuit_str = {}
for fam, circs in fam_circuits.items():
    cl = sorted(circs)
    fam_circuit_str[fam] = f"{len(cl)} circuits: " + ', '.join(cl[:2]) + ('...' if len(cl) > 2 else '')
for fam in set(df_raw['family']) - set(fam_circuits):
    fam_circuit_str[fam] = 'NOT DEFINED YET'

# element → circuit (for flat sheet)
elem_circuit = {}
for _, row in df_circuits_out.iterrows():
    elem_circuit[row['old_name']] = row['circuit']
for _, row in df_quad_circuits.iterrows():
    elem_circuit[row['name']] = row['circuit']
for _, row in df_sext_circuits.iterrows():
    elem_circuit[row['name']] = row['circuit']
df_raw['circuit'] = df_raw['name'].map(elem_circuit).fillna('NOT DEFINED YET')

# enrich catalogue and split_family
catalogue['new_name_type'] = catalogue['family'].map(fam_new_name).fillna('NOT YET DEFINED')
catalogue['regions']       = catalogue['family'].map(fam_regions).fillna('')
catalogue['circuits']      = catalogue['family'].map(fam_circuit_str).fillna('NOT DEFINED YET')
split_family['new_name_type'] = split_family['family'].map(fam_new_name).fillna('NOT YET DEFINED')
split_family['circuits']      = split_family['family'].map(fam_circuit_str).fillna('NOT DEFINED YET')

# ══════════════════════════════════════════════════════════════════════════════
# THIN HIGHER-ORDER MULTIPOLES  (octupoles and decapoles)
# ══════════════════════════════════════════════════════════════════════════════


high_order_records = []
for name, s, etype in zip(tab.name, tab.s, tab.element_type):
    if etype != 'Multipole':
        continue

    el  = line[name]
    knl = np.asarray(getattr(el, 'knl', []), dtype=float)
    k3l = float(knl[3]) if len(knl) > 3 else 0.0
    k4l = float(knl[4]) if len(knl) > 4 else 0.0

    if abs(k3l) <= 1e-15 and abs(k4l) <= 1e-15:
        continue

    if abs(k3l) > 1e-15 and abs(k4l) > 1e-15:
        multipole_type = 'Octupole + Decapole'
    elif abs(k3l) > 1e-15:
        multipole_type = 'Octupole'
    else:
        multipole_type = 'Decapole'

    high_order_records.append({
        'name':               name,
        'new_name':           master_rm3.get(name, 'NOT YET DEFINED'),
        'family':             re.sub(r'[_\.]?\d+$', '', name).upper(),
        'multipole_type':     multipole_type,
        's':                  float(s),
        'region':             get_region_display(float(s)),
        'length':             float(getattr(el, 'length', 0.0) or 0.0),
        'k3l':                k3l,
        'k4l':                k4l,
    })

df_high_order = pd.DataFrame(high_order_records)
high_order_cols = [
    'name', 'new_name', 'family', 'multipole_type', 's', 'region',
    'length', 'k3l', 'k4l',
]
if df_high_order.empty:
    df_high_order = pd.DataFrame(columns=high_order_cols)
else:
    df_high_order = df_high_order[high_order_cols].sort_values('s').reset_index(drop=True)

high_order_summary = df_high_order.groupby(
    ['family', 'multipole_type'], sort=False
).agg(
    count  = ('name',   'count'),
    k3l    = ('k3l',    'mean'),
    k4l    = ('k4l',    'mean'),
    length = ('length', 'mean'),
).reset_index()

total_octupoles = int(df_high_order['multipole_type'].str.contains('Octupole').sum())
total_decapoles = int(df_high_order['multipole_type'].str.contains('Decapole').sum())

# ══════════════════════════════════════════════════════════════════════════════
# EXCEL REPORT
# ══════════════════════════════════════════════════════════════════════════════

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

excel_file = PROJECT_DIR / f'FCC_Magnet_Report_{mode}.xlsx'

# ── Colours ───────────────────────────────────────────────────────────────────
C_NAVY='1A1A5E'; C_BLUE='2E86C1'; C_GREEN='1E8449'; C_GOLD='B7950B'
C_RED='C0392B';  C_PURPLE='6C3483'
C_LBLUE='D4E6F1'; C_LGREEN='D5F5E3'; C_LYELLOW='FEF9E7'
C_LPURPLE='F5EEF8'; C_LRED='FADBD8'; C_LGREY='F2F3F4'; C_LCYAN='D6EAF8'

TYPE_HDR  = {
    'Quadrupole':C_BLUE, 'RBend':C_GREEN, 'Sextupole':C_GOLD,
    'Multipole':C_PURPLE, 'BPM':C_BLUE,
    'Horizontal Corrector':C_RED, 'Vertical Corrector':C_PURPLE,
}
TYPE_FILL = {
    'Quadrupole':C_LBLUE, 'RBend':C_LGREEN, 'Sextupole':C_LYELLOW,
    'Multipole':C_LPURPLE, 'BPM':C_LCYAN,
    'Horizontal Corrector':C_LRED, 'Vertical Corrector':C_LPURPLE,
}

FMT_INT='0'; FMT_F2='0.00'; FMT_F3='0.000'; FMT_F4='0.0000'; FMT_F6='0.000000'

# ── Style helpers ─────────────────────────────────────────────────────────────
def xfill(h):    return PatternFill('solid', start_color=h, fgColor=h)
def xborder(c='CCCCCC'):
    s = Side(style='thin', color=c)
    return Border(left=s, right=s, top=s, bottom=s)
def xhfont(c='FFFFFF', sz=9, bold=True): return Font(name='Arial', bold=bold, color=c, size=sz)
def xcfont(sz=8, bold=False, c='000000'): return Font(name='Arial', size=sz, bold=bold, color=c)
def xctr():      return Alignment(horizontal='center', vertical='center', wrap_text=True)
def xlft():      return Alignment(horizontal='left',   vertical='center', wrap_text=False)
XBORDER = xborder()

def write_hdr(ws, row, headers, bg=C_NAVY, height=28):
    for c, (_, v) in enumerate(headers.items(), 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = xhfont(); cell.fill = xfill(bg)
        cell.alignment = xctr(); cell.border = XBORDER
    ws.row_dimensions[row].height = height

def write_row(ws, row, vals, rf=None, fmts=None):
    f = xfill(rf) if rf else None
    for c, v in enumerate(vals, 1):
        cell = ws.cell(row=row, column=c,
                       value=v if not (isinstance(v, float) and np.isnan(v)) else None)
        cell.font = xcfont(); cell.border = XBORDER; cell.alignment = xlft()
        if f: cell.fill = f
        if fmts and c in fmts: cell.number_format = fmts[c]

def sec_hdr(ws, row, ncols, text, bg=C_NAVY, height=16):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    c = ws.cell(row=row, column=1, value=text)
    c.font = xhfont(sz=9); c.fill = xfill(bg); c.border = XBORDER
    c.alignment = Alignment(horizontal='left', vertical='center')
    ws.row_dimensions[row].height = height

def title_row(ws, row, ncols, text, bg=C_NAVY, height=32):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    c = ws.cell(row=row, column=1, value=text)
    c.font = xhfont(sz=12); c.fill = xfill(bg); c.alignment = xctr()
    ws.row_dimensions[row].height = height

def set_w(ws, widths):
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

def freeze_filter(ws, freeze='A3', ref=None):
    ws.freeze_panes = freeze
    if ref: ws.auto_filter.ref = ref

# ══════════════════════════════════════════════════════════════════════════════
wb = Workbook()
wb.remove(wb.active)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 1 — COVER
# ══════════════════════════════════════════════════════════════════════════════
ws = wb.create_sheet('Cover')
ws.sheet_view.showGridLines = False
title_row(ws, 1, 6, 'FCC-ee Magnet & Circuit Report', height=40)
ws.merge_cells('A2:F2')
c = ws['A2']
c.value = (f'Lattice: {lattice_model}  |  Energy: {mode.upper()}  |  '
           f'Brho = {brho:.4f} T·m  |  Ref. radius = {reference_radius*1000:.0f} mm')
c.font = xhfont(sz=10); c.fill = xfill(C_BLUE); c.alignment = xctr()
ws.row_dimensions[2].height = 18

params = [
    ('Machine',            'FCC-ee'),
    ('Lattice model',      lattice_model),
    ('Energy mode',        mode.upper()),
    ('Circumference [m]',  f'{twissz.circumference:.4f}'),
    ('Brho [T·m]',         f'{brho:.6f}'),
    ('Reference radius [m]', f'{reference_radius:.4f}'),
    ('Qx',                 f'{twissz.qx:.6f}'),
    ('Qy',                 f'{twissz.qy:.6f}'),
    ("Qx'",                f'{twissz.dqx:.4f}'),
    ("Qy'",                f'{twissz.dqy:.4f}'),
    ('Total dipoles',      str(int(total_dipoles))),
    ('Total quadrupoles',  str(int(total_quadrupoles))),
    ('Total sextupoles',   str(int(total_sextupoles))),
    ('Thin octupoles',     str(total_octupoles)),
    ('Thin decapoles',     str(total_decapoles)),
    ('Beam-position monitors', str(total_bpms)),
    ('Horizontal correctors',  str(total_hcorrectors)),
    ('Vertical correctors',    str(total_vcorrectors)),
]
r = 4
ws.cell(r, 1, 'Machine Parameters').font = xhfont(c='000000', sz=10)
ws.row_dimensions[r].height = 16; r += 1
for k, v in params:
    ws.cell(r, 1, k).font = xcfont(bold=True); ws.cell(r, 1).fill = xfill(C_LGREY)
    ws.cell(r, 1).border = XBORDER; ws.cell(r, 2, v).border = XBORDER
    ws.row_dimensions[r].height = 14; r += 1
r += 1
ws.cell(r, 1, 'Sheet Index').font = xhfont(c='000000', sz=10)
ws.row_dimensions[r].height = 16; r += 1
sheets_index = [
    ('1. Cover',            'Machine parameters and sheet index'),
    ('2. Summary_All',      'All magnet families — new names, counts, fields, gradients, circuits, regions'),
    ('3. Summary_Dipoles',  'Dipole families — full info incl. circuits, regions, fields'),
    ('4. Summary_Quads',    'Quadrupole families — full info incl. circuits, regions, gradients'),
    ('5. Summary_Sexts',    'Sextupole/multipole families — full info incl. circuits, regions'),
    ('6. By_Region',        'Family summary per region with avg β-functions and circuits'),
    ('7. Elements_Flat',    'Every element: new name, old name, circuit, fields, β-functions'),
    ('8. Circuits_Dipoles', 'Main dipole circuits (DS+ARC+DS) for power converter engineers'),
    ('9. Circuits_Quads',   'Arc quadrupole circuits (QF/QD, left/right) for power converters'),
    ('10. Circuits_Sexts',  'Arc sextupole circuits for power converters'),
    ('11. Indiv_Powered',   'Individually powered magnets not in main arc circuits'),
    ('12. Higher_Multipoles','Thin octupoles and decapoles — family summary and complete element list'),
    ('13. Correctors_BPMs',  'Corrector/BPM family summary and combined cell-attribution list'),
]
for s_name, desc in sheets_index:
    ws.cell(r, 1, s_name).font = xcfont(bold=True, c=C_BLUE); ws.cell(r, 1).border = XBORDER
    ws.cell(r, 2, desc).border = XBORDER; ws.row_dimensions[r].height = 13; r += 1
ws.column_dimensions['A'].width = 28; ws.column_dimensions['B'].width = 70

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY SHEET BUILDER  (sheets 2-5)
# ══════════════════════════════════════════════════════════════════════════════

SUM_HDRS = {
    'new_name_type': 'New Name\n(type)',
    'family':        'Family',
    'element_type':  'Element\nType',
    'count':         'Count\n(#)',
    'length [m]':    'Length\n[m]',
    'angle [rad]':   'Angle\n[rad]',
    'bfield [T]':    'B-field\n[T]',
    'k1 [1/m2]':     'k1\n[1/m²]',
    'qgrad [T/m]':   'Gradient\n[T/m]',
    'qfield [T]':    'Field@35mm\n[T]',
    'k2 [1/m3]':     'k2\n[1/m³]',
    'sgrad [T/m2]':  'Sext Grad\n[T/m²]',
    'sfield [T]':    'Sext Field\n[T]',
    'betx_mean [m]': 'βx mean\n[m]',
    'bety_mean [m]': 'βy mean\n[m]',
    'circuits':      'Circuits',
    'regions':       'Regions',
    'comment':       'Comment',
}
NCOLS_SUM = len(SUM_HDRS)
SUM_FMTS  = {4:FMT_INT, 5:FMT_F3, 6:FMT_F6, 7:FMT_F3, 8:FMT_F6, 9:FMT_F3,
             10:FMT_F4, 11:FMT_F6, 12:FMT_F3, 13:FMT_F4, 14:FMT_F2, 15:FMT_F2}
SUM_WIDTHS = {'A':14,'B':13,'C':12,'D':7,'E':8,'F':9,'G':8,'H':9,'I':9,
              'J':9,'K':9,'L':9,'M':9,'N':8,'O':8,'P':40,'Q':55,'R':40}

def cat_row_vals(r_data):
    fam = r_data['family']
    return [
        fam_new_name.get(fam, 'NOT YET DEFINED'),
        fam,
        r_data['element_type'],
        r_data['count'],
        r_data.get('length [m]',    np.nan),
        r_data.get('angle [rad]',   np.nan),
        r_data.get('bfield [T]',    np.nan),
        r_data.get('k1 [1/m2]',     np.nan),
        r_data.get('qgrad [T/m]',   np.nan),
        r_data.get('qfield [T]',    np.nan),
        r_data.get('k2 [1/m3]',     np.nan),
        r_data.get('sgrad [T/m2]',  np.nan),
        r_data.get('sfield [T]',    np.nan),
        r_data.get('betx_mean [m]', np.nan),
        r_data.get('bety_mean [m]', np.nan),
        fam_circuit_str.get(fam, 'NOT DEFINED YET'),
        fam_regions.get(fam, ''),
        r_data.get('comment', ''),
    ]

def write_summary_sheet(ws, df_cat, sheet_title, types=None):
    ws.sheet_view.showGridLines = False
    title_row(ws, 1, NCOLS_SUM, sheet_title)
    write_hdr(ws, 2, SUM_HDRS)
    row = 3
    etypes = types if types else ['RBend', 'Quadrupole', 'Sextupole', 'Multipole']
    for et in etypes:
        sub = df_cat[df_cat['element_type'] == et]
        if len(sub) == 0: continue
        sec_hdr(ws, row, NCOLS_SUM,
                f'── {et.upper()} ({len(sub)} families, {int(sub["count"].sum())} elements) ──',
                bg=TYPE_HDR.get(et, C_NAVY))
        row += 1
        rf = TYPE_FILL.get(et, C_LGREY)
        for _, r_data in sub.iterrows():
            write_row(ws, row, cat_row_vals(r_data), rf=rf, fmts=SUM_FMTS)
            row += 1
        row += 1
    set_w(ws, SUM_WIDTHS)
    freeze_filter(ws, 'A3', f'A2:{get_column_letter(NCOLS_SUM)}2')

ws2 = wb.create_sheet('Summary_All')
write_summary_sheet(ws2, catalogue, 'FCC-ee — All Magnet Families')

ws3 = wb.create_sheet('Summary_Dipoles')
write_summary_sheet(ws3, catalogue[catalogue['element_type'] == 'RBend'],
                    'FCC-ee — Dipole Families', ['RBend'])

ws4 = wb.create_sheet('Summary_Quads')
write_summary_sheet(ws4, catalogue[catalogue['element_type'] == 'Quadrupole'],
                    'FCC-ee — Quadrupole Families', ['Quadrupole'])

ws5 = wb.create_sheet('Summary_Sexts')
write_summary_sheet(ws5,
                    catalogue[catalogue['element_type'].isin(['Sextupole', 'Multipole'])],
                    'FCC-ee — Sextupole & Multipole Families',
                    ['Sextupole', 'Multipole'])

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 6 — BY REGION
# ══════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet('By_Region')
ws6.sheet_view.showGridLines = False
REG_HDRS = {
    'region':        'Region',
    'new_name_type': 'New Name\n(type)',
    'family':        'Family',
    'element_type':  'Type',
    'count':         'Count',
    'length [m]':    'Length\n[m]',
    'bfield [T]':    'B-field\n[T]',
    'qgrad [T/m]':   'Gradient\n[T/m]',
    'qfield [T]':    'Field@35mm\n[T]',
    'sgrad [T/m2]':  'Sext Grad\n[T/m²]',
    'sfield [T]':    'Sext Field\n[T]',
    'betx_mean [m]': 'βx mean\n[m]',
    'bety_mean [m]': 'βy mean\n[m]',
    'circuits':      'Circuits',
    'comment':       'Comment',
}
NC6 = len(REG_HDRS)
FMT6 = {5:FMT_INT, 6:FMT_F3, 7:FMT_F3, 8:FMT_F3, 9:FMT_F4, 10:FMT_F3,
        11:FMT_F4, 12:FMT_F2, 13:FMT_F2}
title_row(ws6, 1, NC6, 'FCC-ee — Magnet Families by Region')
write_hdr(ws6, 2, REG_HDRS)
row = 3

def reg_sort(r):
    if r.startswith('ARC'): return (0, r)
    if r.startswith('DS'):  return (1, r)
    return (2, r)

for reg in sorted(split_family['region'].unique(), key=reg_sort):
    grp = split_family[split_family['region'] == reg]
    sec_hdr(ws6, row, NC6,
            f'── {reg}  ({int(grp["count"].sum())} elements) ──', bg=C_NAVY)
    row += 1
    for et in ['RBend', 'Quadrupole', 'Sextupole', 'Multipole']:
        sub = grp[grp['element_type'] == et]
        if len(sub) == 0: continue
        rf = TYPE_FILL.get(et, C_LGREY)
        for _, r_data in sub.iterrows():
            fam = r_data['family']
            vals = [
                reg,
                fam_new_name.get(fam, '?'),
                fam,
                r_data['element_type'],
                r_data['count'],
                r_data.get('length [m]',    np.nan),
                r_data.get('bfield [T]',    np.nan),
                r_data.get('qgrad [T/m]',   np.nan),
                r_data.get('qfield [T]',    np.nan),
                r_data.get('sgrad [T/m2]',  np.nan),
                r_data.get('sfield [T]',    np.nan),
                r_data.get('betx_mean [m]', np.nan),
                r_data.get('bety_mean [m]', np.nan),
                fam_circuit_str.get(fam, 'NOT DEFINED YET'),
                r_data.get('comment', ''),
            ]
            write_row(ws6, row, vals, rf=rf, fmts=FMT6)
            row += 1
    row += 1
set_w(ws6, {'A':9,'B':14,'C':13,'D':12,'E':7,'F':8,'G':8,'H':9,'I':9,
            'J':9,'K':9,'L':8,'M':8,'N':38,'O':40})
freeze_filter(ws6, 'A3', f'A2:{get_column_letter(NC6)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 7 — ELEMENTS FLAT
# ══════════════════════════════════════════════════════════════════════════════
ws7 = wb.create_sheet('Elements_Flat')
ws7.sheet_view.showGridLines = False
FLAT_HDRS = {
    'new_name':     'New Name\n(scheme 3)',
    'old_name':     'Old Name\n',
    'region':       'Region',
    'circuit':      'Circuit',
    'element_type': 'Type',
    's [m]':        'B1 s\n[m]',
    'length [m]':   'Length\n[m]',
    'bfield [T]':   'B-field\n[T]',
    'k1 [1/m2]':    'k1\n[1/m²]',
    'qgrad [T/m]':  'Gradient\n[T/m]',
    'qfield [T]':   'Field@35mm\n[T]',
    'k2 [1/m3]':    'k2\n[1/m³]',
    'sgrad [T/m2]': 'SextGrad\n[T/m²]',
    'sfield [T]':   'SextField\n[T]',
    'betx [m]':     'βx\n[m]',
    'bety [m]':     'βy\n[m]',
    'comment':      'Comment',
    'closest_b2_element':   'B2 element\nname',
    'interbeam_horizontal': 'Centre horizontal\nseparation [m]',
    'interbeam_vertical':   'Centre vertical\nseparation [m]',
    'b1_survey_X':          'B1 survey X\n[m]',
    'b1_survey_Y':          'B1 survey Y\n[m]',
    'b1_survey_Z':          'B1 survey Z\n[m]',
}
NC7 = len(FLAT_HDRS)
FMT7 = {6:FMT_F2, 7:FMT_F3, 8:FMT_F3, 9:FMT_F6, 10:FMT_F3, 11:FMT_F4,
        12:FMT_F6, 13:FMT_F3, 14:FMT_F4, 15:FMT_F2, 16:FMT_F2,
        19:FMT_F3, 20:FMT_F3, 21:FMT_F3, 22:FMT_F3, 23:FMT_F3}
title_row(ws7, 1, NC7, 'FCC-ee — All Magnets Element-by-Element (with Circuits)')
write_hdr(ws7, 2, FLAT_HDRS)
row = 3; prev_reg = None
for _, r_data in df_raw.sort_values('s').iterrows():
    reg  = r_data['region']
    et   = r_data['element_type']
    circ = r_data['circuit']
    if reg != prev_reg:
        sec_hdr(ws7, row, NC7, f'── {reg} ──', bg=TYPE_HDR.get(et, C_NAVY))
        row += 1; prev_reg = reg

    rf = TYPE_FILL.get(et, C_LGREY)
    vals = [
        r_data['new_name'], r_data['name'], reg, circ, et,
        r_data['s'],        r_data['length'],
        r_data.get('bfield',  np.nan), r_data.get('k1',    np.nan),
        r_data.get('qgrad',   np.nan), r_data.get('qfield', np.nan),
        r_data.get('k2',      np.nan), r_data.get('sgrad',  np.nan),
        r_data.get('sfield',  np.nan), r_data.get('betx',   np.nan),
        r_data.get('bety',    np.nan), r_data.get('comment',''),
        r_data.get('closest_b2_element', ''),
        r_data.get('interbeam_horizontal', np.nan),
        r_data.get('interbeam_vertical', np.nan),
        r_data.get('b1_survey_X', np.nan),
        r_data.get('b1_survey_Y', np.nan),
        r_data.get('b1_survey_Z', np.nan),
    ]
    write_row(ws7, row, vals, rf=rf, fmts=FMT7); row += 1
set_w(ws7, {'A':22,'B':22,'C':9,'D':30,'E':12,'F':9,'G':8,'H':8,'I':9,
            'J':9,'K':9,'L':9,'M':9,'N':9,'O':8,'P':8,'Q':35,
            'R':24,'S':18,'T':18,'U':14,'V':14,'W':14})
freeze_filter(ws7, 'A3', f'A2:{get_column_letter(NC7)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 8 — DIPOLE CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════
ws8 = wb.create_sheet('Circuits_Dipoles')
ws8.sheet_view.showGridLines = False
DIP_HDRS = {
    'circuit':      'Circuit',
    'arc_sector':   'Arc Sector',
    'new_name':     'New Name',
    'old_name':     'Old Name\n',
    'family':       'Family',
    'region':       'Region',
    's [m]':        's [m]',
    'length [m]':   'Length\n[m]',
    'angle [rad]':  'Angle\n[rad]',
    'bfield [T]':   'B-field\n[T]',
    'betx [m]':     'βx [m]',
    'bety [m]':     'βy [m]',
    'comment':      'Comment',
}
NC8 = len(DIP_HDRS)
FMT8 = {7:FMT_F2, 8:FMT_F3, 9:FMT_F6, 10:FMT_F3, 11:FMT_F2, 12:FMT_F2}
title_row(ws8, 1, NC8, 'FCC-ee — Main Dipole Circuits  (DS + ARC + DS per sector)')
write_hdr(ws8, 2, DIP_HDRS)
row = 3
for cname, grp in df_circuits_out.groupby('circuit', sort=False):
    r0 = grp.iloc[0]
    tot_ang = grp['angle [rad]'].sum()
    sec_hdr(ws8, row, NC8,
            f'── {cname}  |  {len(grp)} dipoles  |  '
            f'Total angle: {tot_ang:.6f} rad  |  '
            f'{r0["from_ds"]} → {r0["arc"]} → {r0["to_ds"]}', bg=C_GREEN)
    row += 1
    for _, rd in grp.iterrows():
        write_row(ws8, row, [
            cname, rd.get('arc',''), rd['new_name'], rd['old_name'],
            rd['family'], rd['region'], rd['s [m]'], rd['length [m]'],
            rd['angle [rad]'], rd['bfield [T]'], rd['betx [m]'], rd['bety [m]'],
            rd.get('comment',''),
        ], rf=C_LGREEN, fmts=FMT8)
        row += 1
    row += 1
set_w(ws8, {'A':28,'B':18,'C':18,'D':20,'E':12,'F':9,
            'G':9,'H':8,'I':9,'J':8,'K':8,'L':8,'M':40})
freeze_filter(ws8, 'A3', f'A2:{get_column_letter(NC8)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 9 — QUADRUPOLE CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════
ws9 = wb.create_sheet('Circuits_Quads')
ws9.sheet_view.showGridLines = False
QUAD_HDRS = {
    'circuit':   'Circuit',
    'polarity':  'Polarity',
    'side':      'Side',
    'new_name':  'New Name',
    'name':      'Old Name\n',
    'family':    'Family',
    'region':    'Region',
    's':         's [m]',
    'length':    'Length\n[m]',
    'k1':        'k1\n[1/m²]',
    'qgrad':     'Gradient\n[T/m]',
    'qfield':    'Field@35mm\n[T]',
    'betx':      'βx [m]',
    'bety':      'βy [m]',
    'comment':   'Comment',
}
NC9 = len(QUAD_HDRS)
FMT9 = {8:FMT_F2, 9:FMT_F3, 10:FMT_F6, 11:FMT_F3, 12:FMT_F4, 13:FMT_F2, 14:FMT_F2}
title_row(ws9, 1, NC9, 'FCC-ee — Arc Quadrupole Circuits  (focusing/defocusing, left/right half)')
write_hdr(ws9, 2, QUAD_HDRS)
row = 3
for cname, grp in df_quad_circuits.groupby('circuit', sort=False):
    pol  = grp['polarity'].iloc[0]
    side = grp['side'].iloc[0]
    pol_str  = 'Focusing' if pol == 'qf' else 'Defocusing'
    side_str = 'Left half' if side == 'l' else 'Right half'
    bg = C_BLUE if pol == 'qf' else C_PURPLE
    rf = C_LBLUE if pol == 'qf' else C_LPURPLE
    sec_hdr(ws9, row, NC9,
            f'── {cname}  |  {len(grp)} quads  |  {pol_str}  |  {side_str}  |  '
            f'k1 range: [{grp["k1"].min():.5f}, {grp["k1"].max():.5f}] m⁻²', bg=bg)
    row += 1
    for _, rd in grp.iterrows():
        write_row(ws9, row, [
            cname, pol_str, side_str, rd['new_name'], rd['name'],
            rd['family'], rd['region'], rd['s'], rd['length'],
            rd['k1'], rd['qgrad'], rd['qfield'], rd['betx'], rd['bety'],
            rd.get('comment',''),
        ], rf=rf, fmts=FMT9)
        row += 1
    row += 1
set_w(ws9, {'A':30,'B':11,'C':10,'D':18,'E':18,'F':11,'G':9,
            'H':9,'I':8,'J':9,'K':9,'L':9,'M':8,'N':8,'O':35})
freeze_filter(ws9, 'A3', f'A2:{get_column_letter(NC9)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 10 — SEXTUPOLE CIRCUITS
# ══════════════════════════════════════════════════════════════════════════════
ws10 = wb.create_sheet('Circuits_Sexts')
ws10.sheet_view.showGridLines = False
SEXT_HDRS = {
    'circuit':      'Circuit',
    'family_label': 'Family\nLabel',
    'side':         'Side',
    'new_name':     'New Name',
    'name':         'Old Name\n',
    'family':       'Family',
    'region':       'Region',
    's':            's [m]',
    'length':       'Length\n[m]',
    'k2':           'k2\n[1/m³]',
    'sgrad':        'Sext Grad\n[T/m²]',
    'sfield':       'SextField\n[T]',
    'betx':         'βx [m]',
    'bety':         'βy [m]',
    'comment':      'Comment',
}
NC10 = len(SEXT_HDRS)
FMT10 = {8:FMT_F2, 9:FMT_F3, 10:FMT_F6, 11:FMT_F3, 12:FMT_F4, 13:FMT_F2, 14:FMT_F2}
title_row(ws10, 1, NC10,
          'FCC-ee — Arc Sextupole Circuits  (SF1A/SF2A/SD1A/SD2A, left/right half)')
write_hdr(ws10, 2, SEXT_HDRS)
row = 3
for cname, grp in df_sext_circuits.groupby('circuit', sort=False):
    flab     = grp['family_label'].iloc[0]
    side     = grp['side'].iloc[0]
    foc      = flab in ('sf1a', 'sf2a')
    side_str = 'Left' if side == 'l' else 'Right'
    sec_hdr(ws10, row, NC10,
            f'── {cname}  |  {len(grp)} sextupoles  |  '
            f'{"Focusing" if foc else "Defocusing"} ({flab.upper()})  |  {side_str}  |  '
            f'k2 range: [{grp["k2"].min():.4f}, {grp["k2"].max():.4f}] m⁻³', bg=C_GOLD)
    row += 1
    for _, rd in grp.iterrows():
        write_row(ws10, row, [
            cname, flab.upper(), side_str, rd['new_name'], rd['name'],
            rd['family'], rd['region'], rd['s'], rd['length'],
            rd['k2'], rd['sgrad'], rd['sfield'], rd['betx'], rd['bety'],
            rd.get('comment',''),
        ], rf=C_LYELLOW, fmts=FMT10)
        row += 1
    row += 1
set_w(ws10, {'A':30,'B':10,'C':7,'D':18,'E':18,'F':11,'G':9,
             'H':9,'I':8,'J':9,'K':9,'L':9,'M':8,'N':8,'O':35})
freeze_filter(ws10, 'A3', f'A2:{get_column_letter(NC10)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 11 — INDIVIDUALLY POWERED
# ══════════════════════════════════════════════════════════════════════════════
ws11 = wb.create_sheet('Indiv_Powered')
ws11.sheet_view.showGridLines = False
INDIV_HDRS = {
    'new_name':     'New Name',
    'old_name':     'Old Name\n',
    'family':       'Family',
    'element_type': 'Type',
    'region':       'Region',
    's':            's [m]',
    'length':       'Length\n[m]',
    'bfield':       'B-field\n[T]',
    'qgrad':        'Gradient\n[T/m]',
    'qfield':       'Field@35mm\n[T]',
    'sgrad':        'SextGrad\n[T/m²]',
    'sfield':       'SextField\n[T]',
    'betx':         'βx [m]',
    'bety':         'βy [m]',
    'comment':      'Comment',
}
NC11 = len(INDIV_HDRS)
FMT11 = {6:FMT_F2, 7:FMT_F3, 8:FMT_F3, 9:FMT_F3, 10:FMT_F4,
         11:FMT_F3, 12:FMT_F4, 13:FMT_F2, 14:FMT_F2}
title_row(ws11, 1, NC11,
          'FCC-ee — Individually Powered Magnets (not in main arc circuits)')
write_hdr(ws11, 2, INDIV_HDRS)
row = 3
for label, df_ind, et in [
    ('DIPOLES',     df_dip_remaining,  'RBend'),
    ('QUADRUPOLES', df_quad_remaining, 'Quadrupole'),
    ('SEXTUPOLES',  df_sext_remaining, 'Sextupole'),
]:
    if len(df_ind) == 0: continue
    sec_hdr(ws11, row, NC11,
            f'── {label}  ({len(df_ind)} elements) ──',
            bg=TYPE_HDR.get(et, C_RED))
    row += 1
    rf = TYPE_FILL.get(et, C_LGREY)
    for _, rd in df_ind.sort_values('s').iterrows():
        write_row(ws11, row, [
            rd.get('new_name',''), rd['name'], rd['family'], rd['element_type'],
            rd.get('region',''), rd['s'], rd['length'],
            rd.get('bfield',  np.nan), rd.get('qgrad',  np.nan),
            rd.get('qfield',  np.nan), rd.get('sgrad',  np.nan),
            rd.get('sfield',  np.nan), rd.get('betx',   np.nan),
            rd.get('bety',    np.nan), rd.get('comment',''),
        ], rf=rf, fmts=FMT11)
        row += 1
    row += 1
set_w(ws11, {'A':22,'B':22,'C':13,'D':12,'E':9,'F':9,'G':8,'H':8,'I':9,
             'J':9,'K':9,'L':9,'M':8,'N':8,'O':35})
freeze_filter(ws11, 'A3', f'A2:{get_column_letter(NC11)}2')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 12 — THIN OCTUPOLES AND DECAPOLES
# ══════════════════════════════════════════════════════════════════════════════
ws12 = wb.create_sheet('Higher_Multipoles')
ws12.sheet_view.showGridLines = False

HOM_HDRS = {
    'family':         'Family',
    'multipole_type': 'Multipole\nType',
    'count':          'Count',
    'k3l':            'k3L\n[1/m³]',
    'k4l':            'k4L\n[1/m⁴]',
    'length':         'Length\n[m]',
}
NC12S = len(HOM_HDRS)
NC12_TOTAL = 9
FMT12S = {3:FMT_INT, 4:FMT_F6, 5:FMT_F6, 6:FMT_F3}

title_row(ws12, 1, NC12_TOTAL,
          'FCC-ee — Thin Higher-Order Multipoles (Octupoles and Decapoles)')
sec_hdr(ws12, 2, NC12_TOTAL,
        f'FAMILY SUMMARY  |  {total_octupoles} octupoles  |  '
        f'{total_decapoles} decapoles', bg=C_PURPLE, height=20)
write_hdr(ws12, 3, HOM_HDRS, bg=C_PURPLE)

row = 4
for mtype in ['Octupole', 'Decapole', 'Octupole + Decapole']:
    sub = high_order_summary[high_order_summary['multipole_type'] == mtype]
    if len(sub) == 0:
        continue
    for _, rd in sub.iterrows():
        write_row(ws12, row, [
            rd['family'], rd['multipole_type'], rd['count'],
            rd['k3l'], rd['k4l'], rd['length'],
        ], rf=C_LPURPLE, fmts=FMT12S)
        row += 1

row += 2
detail_title_row = row
HOM_DETAIL_HDRS = {
    'new_name':             'New Name\n(scheme 3)',
    'name':                 'Old Name',
    'family':               'Family',
    'multipole_type':       'Multipole\nType',
    'region':               'Region',
    's':                    's\n[m]',
    'length':               'Length\n[m]',
    'k3l':                  'k3L\n[1/m³]',
    'k4l':                  'k4L\n[1/m⁴]',
}
NC12D = len(HOM_DETAIL_HDRS)
sec_hdr(ws12, row, NC12D, 'COMPLETE ELEMENT LIST', bg=C_NAVY, height=20)
row += 1
detail_header_row = row
write_hdr(ws12, row, HOM_DETAIL_HDRS)
row += 1

FMT12D = {6:FMT_F2, 7:FMT_F3, 8:FMT_F6, 9:FMT_F6}
for _, rd in df_high_order.iterrows():
    write_row(ws12, row, [
        rd['new_name'], rd['name'], rd['family'], rd['multipole_type'],
        rd['region'], rd['s'], rd['length'], rd['k3l'], rd['k4l'],
    ], rf=C_LPURPLE, fmts=FMT12D)
    row += 1

set_w(ws12, {
    'A':22, 'B':22, 'C':13, 'D':18, 'E':10, 'F':10,
    'G':10, 'H':12, 'I':12,
})
ws12.freeze_panes = f'A{detail_header_row + 1}'
ws12.auto_filter.ref = (
    f'A{detail_header_row}:{get_column_letter(NC12D)}{max(detail_header_row, row - 1)}'
)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 13 — CORRECTORS, BPMS, AND COMPLETE CELL ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
ws13 = wb.create_sheet('Correctors_BPMs')
ws13.sheet_view.showGridLines = False

DEV_SUM_HDRS = {
    'scope':             'Scope',
    'region':            'Region',
    'bpms':              'BPMs',
    'hcorrectors':       'Horizontal\nCorrectors',
    'vcorrectors':       'Vertical\nCorrectors',
    'total':             'Total Devices',
}
NC13S = len(DEV_SUM_HDRS)
NC13D = 15
FMT13S = {3:FMT_INT, 4:FMT_INT, 5:FMT_INT, 6:FMT_INT}

title_row(ws13, 1, NC13D,
          'FCC-ee — Correctors, BPMs, and Complete Cell Attribution')
sec_hdr(
    ws13, 2, NC13D,
    f'RING AND REGION SUMMARY  |  {total_bpms} BPMs  |  '
    f'{total_hcorrectors} horizontal correctors  |  '
    f'{total_vcorrectors} vertical correctors',
    bg=C_BLUE, height=20,
)
write_hdr(ws13, 3, DEV_SUM_HDRS, bg=C_BLUE)

row = 4
write_row(ws13, row, [
    'Full ring', 'ALL',
    total_bpms, total_hcorrectors, total_vcorrectors,
    total_bpms + total_hcorrectors + total_vcorrectors,
], rf=C_LCYAN, fmts=FMT13S)
row += 1

region_order = [
    region
    for sector in range(1, 9)
    for region in (f'INS{sector}', f'DS{sector}', f'ARC{sector}')
]
for region in region_order:
    if region not in device_region_summary.index:
        continue
    counts = device_region_summary.loc[region]
    nbpm = int(counts.get('BPM', 0))
    nh = int(counts.get('Horizontal Corrector', 0))
    nv = int(counts.get('Vertical Corrector', 0))
    write_row(ws13, row, [
        'Region', region, nbpm, nh, nv, nbpm + nh + nv,
    ], rf=C_LGREY, fmts=FMT13S)
    row += 1

# Build a combined position-sorted list.  It contains every Elements_Flat row
# plus every BPM and horizontal/vertical corrector.
quad_lookup = [
    (rd['name'], float(rd['s']), rd['new_name'])
    for _, rd in df_raw[df_raw['element_type'] == 'Quadrupole'].iterrows()
]
quad_positions = [(q[0], q[1]) for q in quad_lookup]

combined_records = []
for _, rd in df_raw.iterrows():
    cell, side, sector = cell_parts(rd['new_name'])
    assoc_quad = ''
    quad_distance = np.nan
    if rd['element_type'] == 'Quadrupole':
        assoc_quad = rd['name']
        quad_distance = 0.0
        attribution = 'Quadrupole reference'
    elif rd['element_type'] == 'Sextupole':
        candidate = nearest_quad(float(rd['s']), quad_positions)
        if (candidate is not None and
                circular_dist(float(rd['s']), candidate[1], circumference)
                <= S3_SAME_GIRDER_MAX_DIST + 1e-6):
            assoc_quad = candidate[0]
            quad_distance = circular_dist(float(rd['s']), candidate[1], circumference)
            attribution = 'Nearest quadrupole — same girder'
        else:
            attribution = 'Scheme 3 longitudinal attribution'
    else:
        attribution = 'Scheme 3 longitudinal attribution'

    combined_records.append({
        'new_name': rd['new_name'],
        'name': rd['name'],
        'family': rd['family'],
        'device_type': rd['element_type'],
        'region': rd['region'],
        's': rd['s'],
        'length': rd['length'],
        'cell': cell,
        'side': side,
        'sector': sector,
        'cell_attribution': attribution,
        'associated_quad': assoc_quad,
        'quad_distance': quad_distance,
        'circuit': rd.get('circuit', ''),
        'k0l': np.nan,
        'k0sl': np.nan,
        'comment': rd.get('comment', ''),
    })

for _, rd in df_devices.iterrows():
    combined_records.append({
        'new_name': rd['new_name'],
        'name': rd['name'],
        'family': rd['family'],
        'device_type': rd['device_type'],
        'region': rd['region'],
        's': rd['s'],
        'length': rd['length'],
        'cell': rd['cell'],
        'side': rd['side'],
        'sector': rd['sector'],
        'cell_attribution': rd['cell_attribution'],
        'associated_quad': rd['associated_quad'],
        'quad_distance': rd['quad_distance'],
        'circuit': '',
        'k0l': rd['k0l'],
        'k0sl': rd['k0sl'],
        'comment': rd['comment'],
    })

df_combined = pd.DataFrame(combined_records)
detail_order = {
    'Quadrupole': 0, 'BPM': 1, 'Horizontal Corrector': 2,
    'Vertical Corrector': 3, 'Sextupole': 4, 'RBend': 5,
}
df_combined['_order'] = df_combined['device_type'].map(detail_order).fillna(9)
df_combined = df_combined.sort_values(
    ['s', '_order', 'name']
).drop(columns='_order').reset_index(drop=True)

row += 2
detail_title_row_13 = row
DEV_DETAIL_HDRS = {
    'new_name':          'New Name\n(scheme 3)',
    'name':              'Old Name',
    'family':            'Family',
    'device_type':       'Type',
    'region':            'Region',
    's':                 's\n[m]',
    'length':            'Length\n[m]',
    'cell':              'Cell',
    'side':              'Side',
    'sector':            'Sector',
    'associated_quad':   'Associated Quadrupole',
    'quad_distance':     'Distance to Quad\n[m]',
    'circuit':           'Circuit',
    'k0l':               'k0L\n[rad]',
    'k0sl':              'k0sL\n[rad]',
}
sec_hdr(ws13, row, NC13D,
        f'COMPLETE ELEMENT LIST  |  {len(df_combined)} elements',
        bg=C_NAVY, height=20)
row += 1
detail_header_row_13 = row
write_hdr(ws13, row, DEV_DETAIL_HDRS)
row += 1

FMT13D = {
    6:FMT_F2, 7:FMT_F3, 8:FMT_INT, 10:FMT_INT,
    12:FMT_F3, 14:FMT_F6, 15:FMT_F6,
}
for _, rd in df_combined.iterrows():
    dtype = rd['device_type']
    write_row(ws13, row, [
        rd['new_name'], rd['name'], rd['family'], dtype, rd['region'],
        rd['s'], rd['length'], rd['cell'], rd['side'], rd['sector'],
        rd['associated_quad'], rd['quad_distance'], rd['circuit'],
        rd['k0l'], rd['k0sl'],
    ], rf=TYPE_FILL.get(dtype, C_LGREY), fmts=FMT13D)
    row += 1

set_w(ws13, {
    'A':22, 'B':24, 'C':18, 'D':20, 'E':9, 'F':10, 'G':10,
    'H':8, 'I':7, 'J':7, 'K':22, 'L':14, 'M':30, 'N':12, 'O':12,
})
# Freeze only the compact ring/region summary header.
ws13.freeze_panes = 'A4'
ws13.auto_filter.ref = (
    f'A{detail_header_row_13}:{get_column_letter(NC13D)}{max(detail_header_row_13, row - 1)}'
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
wb.save(excel_file)
print(f'\nExcel report saved to {excel_file}')
print(f'  Sheets: {[s.title for s in wb.worksheets]}')
sys.exit(0)  # Do not execute the exploratory survey cells appended below.
# %%





#%%



# %%
