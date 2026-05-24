"""
FLARE LP polytope + 5 at.% grid enumeration.

Constraint-first screening of 11-element refractory alloys against FLARE
response thresholds (10x pure W and 10x EUROFER97). The feasible region
is defined as an LP polytope on the atomic-fraction simplex; the 5 at.%
grid is enumerated inside it by LP-guided recursion. No FLARE evaluation
is done per composition.

Patched constraints (mass-extensive rows):
    sum_i x_i * M_i * (A[r,i] - factor * b_ref[r]) <= 0
DPA row stays at-linear:
    sum_i x_i * A[r,i] <= factor * b_ref[r]
"""

import os
import time
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from flare import Alloy
from flare.constants import atomic_masses

# Optional MiKTeX path (only added if it actually exists locally).
for _p in [
    r'C:\Users\ifada\AppData\Local\Programs\MiKTeX\miktex\bin\x64',
    r'C:\Users\ifadasamuel\AppData\Local\Programs\MiKTeX\miktex\bin\x64',
    r'C:\Program Files\MiKTeX\miktex\bin\x64',
]:
    if os.path.isdir(_p):
        os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')
        break

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "text.usetex": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.linewidth": 1.0,
    "axes.formatter.useoffset": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.constrained_layout.use": False,
})

# Parameters
GRID_STEP = 0.05
GRID_UNITS = int(round(1.0 / GRID_STEP))
FACTOR = 10
TOL = 1e-9

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
EUROFER_FILE = os.path.join(BASE_DIR, 'EUROFER97_flare_DS.csv')

# Fixed FLARE conditions
FLUX     = '8.000E+13'
IRR      = '2.000E+00'
DEC_1S   = '3.169E-08'
DEC_1M   = '1.000E-01'
DEC_100Y = '1.000E+02'

# Alloy system
ELEMENTS = ['Ti', 'V', 'Ta', 'Nb', 'Mo', 'Zr', 'Cr', 'Hf', 'Fe', 'Re', 'W']
N = len(ELEMENTS)
M_AT = np.array([atomic_masses[el] for el in ELEMENTS], dtype=float)
DPA_ROW = 2  # only atom-extensive row in A


def _get_response(alloy_obj, dec_time, col):
    """Single FLARE response at one decay time, averaged over spectra."""
    resp = alloy_obj.responses_simple
    f = resp[
        (resp['Flux'] == FLUX) &
        (resp['Irr_time'] == IRR) &
        (resp['Dec_time'] == dec_time)
    ]
    return float(f[col].mean())


def _get_all_responses(alloy_obj, dec_time):
    """Full FLARE response set (detailed + aggregates) at one decay time."""
    rs = alloy_obj.responses_simple
    ra = alloy_obj.responses
    ms = (rs['Flux'] == FLUX) & (rs['Irr_time'] == IRR) & (rs['Dec_time'] == dec_time)
    ma = (ra['Flux'] == FLUX) & (ra['Irr_time'] == IRR) & (ra['Dec_time'] == dec_time)
    row_s = rs[ms].mean(numeric_only=True)
    row_a = ra[ma].mean(numeric_only=True)

    r = {}
    for col in ['M_H1', 'M_H2', 'M_H3', 'M_He3', 'M_He4', 'DPA',
                'BQ_A', 'BQ_B', 'BQ_G', 'HT_A', 'HT_B', 'HT_G', 'DOSE']:
        r[col] = float(row_a[col]) if col in row_a.index else float(row_s.get(col, 0))

    r['M_GAS'] = r['M_H1'] + r['M_H2'] + r['M_H3'] + r['M_He3'] + r['M_He4']
    r['BQ_T']  = r['BQ_A'] + r['BQ_B'] + r['BQ_G']
    r['HT_T']  = r['HT_A'] + r['HT_B'] + r['HT_G']
    return r


def build_constraint_matrix():
    """Pure-element FLARE response matrix A (5 x N).

    Rows: M_GAS@1s, HT_T@1s, DPA@1s, DOSE@1m, DOSE@100y.
    """
    A = np.zeros((5, N), dtype=float)
    for j, el in enumerate(ELEMENTS):
        a = Alloy({el: 1.0})
        A[0, j] = _get_response(a, DEC_1S,   'M_GAS')
        A[1, j] = _get_response(a, DEC_1S,   'HT_T')
        A[2, j] = _get_response(a, DEC_1S,   'DPA')
        A[3, j] = _get_response(a, DEC_1M,   'DOSE')
        A[4, j] = _get_response(a, DEC_100Y, 'DOSE')
    return A


def get_reference_vectors():
    """Reference response vectors b_W (pure W) and b_E (EUROFER97 mean)."""
    w = Alloy({'W': 1.0})
    b_W = np.array([
        _get_response(w, DEC_1S,   'M_GAS'),
        _get_response(w, DEC_1S,   'HT_T'),
        _get_response(w, DEC_1S,   'DPA'),
        _get_response(w, DEC_1M,   'DOSE'),
        _get_response(w, DEC_100Y, 'DOSE'),
    ])

    df_e = pd.read_csv(EUROFER_FILE)
    df_e['M_GAS'] = df_e['M_H1'] + df_e['M_H2'] + df_e['M_H3'] + df_e['M_He3'] + df_e['M_He4']
    df_e['HT_T']  = df_e['HT_A'] + df_e['HT_B'] + df_e['HT_G']
    df_e = df_e[(df_e['Flux'] == 8e13) & (df_e['Irr_time'] == 2.0)]

    def _get_e(dec_time, col):
        row = df_e[np.isclose(df_e['Dec_time'].astype(float), float(dec_time), atol=1e-12)].mean(numeric_only=True)
        return float(row[col])

    b_E = np.array([
        _get_e(3.169e-08, 'M_GAS'),
        _get_e(3.169e-08, 'HT_T'),
        _get_e(3.169e-08, 'DPA'),
        _get_e(1.0e-01,   'DOSE'),
        _get_e(1.0e+02,   'DOSE'),
    ])
    return b_W, b_E


def _build_flare_constraint_rows(A, b_ref, factor=FACTOR, M=M_AT, dpa_row=DPA_ROW):
    """Patched LP rows (G_flare, h_flare).

    Mass-extensive rows:  M_i * (A[r,i] - factor * b_ref[r]) <= 0
    DPA row:              A[r,i]                              <= factor * b_ref[r]
    """
    G_flare = np.zeros_like(A, dtype=float)
    h_flare = np.zeros(A.shape[0], dtype=float)
    for r in range(A.shape[0]):
        if r == dpa_row:
            G_flare[r] = A[r]
            h_flare[r] = factor * b_ref[r]
        else:
            G_flare[r] = M * (A[r] - factor * b_ref[r])
            h_flare[r] = 0.0
    return G_flare, h_flare


def build_full_inequality_system(A_flare, b_flare, lower_bounds=None, upper_bounds=None):
    """Stack FLARE rows with bounds into one Gx <= h system.

    The simplex equality sum(x) = 1 is handled separately by callers.
    """
    n_vars = A_flare.shape[1]
    if lower_bounds is None:
        lower_bounds = np.zeros(n_vars)
    if upper_bounds is None:
        upper_bounds = np.ones(n_vars)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    G = np.vstack([A_flare, -np.eye(n_vars), np.eye(n_vars)])
    h = np.concatenate([np.asarray(b_flare, dtype=float),
                        -lower_bounds,
                        upper_bounds])
    return G, h


def enumerate_vertices_exact(G, h, tol=TOL):
    """Exact polytope vertices of {Gx <= h, sum(x) = 1}.

    Each vertex is the intersection of the simplex equality with N-1
    active inequality rows.
    """
    n_vars = G.shape[1]
    vertices = []
    Aeq = np.ones((1, n_vars), dtype=float)
    beq = np.array([1.0], dtype=float)

    for active in itertools.combinations(range(G.shape[0]), n_vars - 1):
        M = np.vstack([Aeq, G[list(active)]])
        rhs = np.concatenate([beq, h[list(active)]])
        if np.linalg.matrix_rank(M) < n_vars:
            continue
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        if abs(x.sum() - 1.0) > tol:
            continue
        if np.any(G @ x > h + tol):
            continue
        x = np.where(np.abs(x) < tol, 0.0, x)
        x = np.where(np.abs(x - 1.0) < tol, 1.0, x)
        vertices.append(np.round(x, 10))

    if not vertices:
        return np.empty((0, n_vars), dtype=float)
    return np.unique(np.array(vertices), axis=0)


def sample_polytope_grid_5atpct_all_multicomponent(G, h, grid_step=GRID_STEP, tol=TOL):
    """LP-guided enumeration of all 5 at.% grid points inside {Gx <= h, sum(x) = 1}.

    At each recursion depth two small LPs give the feasible integer-units
    range for the next variable, so only branches with feasible completions
    are explored. Pure-element points are kept.
    """
    if not np.isclose(round(1.0 / grid_step) * grid_step, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"grid_step={grid_step} must evenly divide 1.0")

    units_total = round(1.0 / grid_step)
    n_vars = G.shape[1]
    feasible = []
    x_units = np.zeros(n_vars, dtype=int)

    @lru_cache(maxsize=None)
    def lp_interval(pos, units_left, slack_key):
        slack = np.array(slack_key, dtype=float)
        remaining = n_vars - pos
        remaining_sum = units_left / units_total
        G_sub = G[:, pos:]
        c = np.zeros(remaining, dtype=float); c[0] = 1.0
        bounds = [(0.0, remaining_sum)] * remaining
        A_eq = np.ones((1, remaining), dtype=float)
        b_eq = np.array([remaining_sum], dtype=float)

        mn = linprog(c,  A_ub=G_sub, b_ub=slack, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not mn.success:
            return None
        mx = linprog(-c, A_ub=G_sub, b_ub=slack, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not mx.success:
            return None

        lo = max(0,         int(np.ceil((mn.fun - tol) * units_total)))
        up = min(units_left, int(np.floor((-mx.fun + tol) * units_total)))
        return (lo, up) if lo <= up else None

    def recurse(pos, units_left, used_resp):
        # No early-exit on partial sums: patched mass-extensive rows are
        # sign-indefinite, so the running sum is non-monotone. lp_interval
        # handles infeasibility exactly.
        if pos == n_vars - 1:
            x_units[pos] = units_left
            final_resp = used_resp + (units_left / units_total) * G[:, pos]
            x = x_units.copy() / units_total
            if np.all(final_resp <= h + tol):
                feasible.append(x)
            return

        slack = h - used_resp
        interval = lp_interval(pos, units_left, tuple(np.round(slack, 12)))
        if interval is None:
            return
        lower, upper = interval
        for u in range(lower, upper + 1):
            x_units[pos] = u
            recurse(pos + 1, units_left - u, used_resp + (u / units_total) * G[:, pos])

    recurse(0, units_total, np.zeros(G.shape[0], dtype=float))
    if not feasible:
        return np.empty((0, n_vars), dtype=float)
    return np.array(feasible, dtype=float)


def evaluate_responses(samples):
    """FLARE responses for the enumerated compositions.

    Mass-extensive responses are wt-linear, DPA is at-linear. The wt
    fractions are computed once per composition from M_AT.

    Column order: M_H1..M_He4, DPA, BQ_A/B/G, HT_A/B/G, M_GAS, BQ_T,
    HT_T, DOSE_1s, DOSE_1m, DOSE_100y.
    """
    detail_cols_1s = [
        "M_H1", "M_H2", "M_H3", "M_He3", "M_He4",
        "BQ_A", "BQ_B", "BQ_G",
        "HT_A", "HT_B", "HT_G",
    ]

    per_el = {c: np.zeros(N, dtype=float) for c in detail_cols_1s}
    dpa = np.zeros(N, dtype=float)
    dose_1s = np.zeros(N, dtype=float)
    dose_1m = np.zeros(N, dtype=float)
    dose_100y = np.zeros(N, dtype=float)

    for j, el in enumerate(ELEMENTS):
        a = Alloy({el: 1.0})
        row_1s   = _get_all_responses(a, DEC_1S)
        row_1m   = _get_all_responses(a, DEC_1M)
        row_100y = _get_all_responses(a, DEC_100Y)
        for c in detail_cols_1s:
            per_el[c][j] = row_1s[c]
        dpa[j]       = row_1s["DPA"]
        dose_1s[j]   = row_1s["DOSE"]
        dose_1m[j]   = row_1m["DOSE"]
        dose_100y[j] = row_100y["DOSE"]

    m_avg = samples @ M_AT
    wt = (samples * M_AT) / m_avg[:, None]

    out = {c: wt @ per_el[c] for c in detail_cols_1s}
    out["DPA"]       = samples @ dpa
    out["M_GAS"]     = out["M_H1"] + out["M_H2"] + out["M_H3"] + out["M_He3"] + out["M_He4"]
    out["BQ_T"]      = out["BQ_A"] + out["BQ_B"] + out["BQ_G"]
    out["HT_T"]      = out["HT_A"] + out["HT_B"] + out["HT_G"]
    out["DOSE_1s"]   = wt @ dose_1s
    out["DOSE_1m"]   = wt @ dose_1m
    out["DOSE_100y"] = wt @ dose_100y

    col_order = [
        "M_H1", "M_H2", "M_H3", "M_He3", "M_He4",
        "DPA",
        "BQ_A", "BQ_B", "BQ_G",
        "HT_A", "HT_B", "HT_G",
        "M_GAS", "BQ_T", "HT_T",
        "DOSE_1s", "DOSE_1m", "DOSE_100y",
    ]
    return pd.DataFrame({c: out[c] for c in col_order})


# ---------- plotting ----------

def _generate_polygon(n, radius=1.0, rotation=np.pi / 180.0):
    """Regular n-gon used as the affine projection scaffold."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _project(samples, polygon):
    """Project compositions onto the polygon (x @ polygon, normalized)."""
    X = samples.copy()
    rs = X.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (X / rs) @ polygon


def _setup_affine_ax(ax, polygon):
    """Background polygon + element labels."""
    ax.add_patch(MplPolygon(polygon, edgecolor='none', facecolor='#4d4d4d', alpha=0.07))
    label_radius = 1.10
    for i, el in enumerate(ELEMENTS):
        angle = (2 * np.pi * i / N) + np.pi / 180.0
        ax.text(label_radius * np.cos(angle), label_radius * np.sin(angle), el,
                fontsize=11, ha='center', va='center',
                fontweight='bold', color='black', clip_on=False)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_feasible_spaces_alloys_AB(vertices_A, samples_A, vertices_B, samples_B, out_dir, out_stub):
    """Two-panel figure: Alloy A (10x W) and Alloy B (10x EUROFER97)."""
    polygon = _generate_polygon(N)
    color_A, color_B = "maroon", "#1F77B4"
    fill_A = mcolors.to_rgba(color_A, alpha=0.16)
    fill_B = mcolors.to_rgba(color_B, alpha=0.16)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.subplots_adjust(top=0.98, bottom=0.20, left=0.04, right=0.97, wspace=0.15)

    panels = [
        (axes[0], vertices_A, samples_A, color_A, fill_A, "A"),
        (axes[1], vertices_B, samples_B, color_B, fill_B, "B"),
    ]

    for ax, vertices, samples, main_color, fill_color, label in panels:
        _setup_affine_ax(ax, polygon)

        proj_v = _project(vertices, polygon) if len(vertices) else np.empty((0, 2))
        hull_pts = np.empty((0, 2))
        hull_idx = np.array([], dtype=int)
        interior_idx = np.array([], dtype=int)
        if len(proj_v) >= 3:
            hull = ConvexHull(proj_v)
            hull_idx = hull.vertices
            hull_pts = np.vstack([proj_v[hull.vertices], proj_v[hull.vertices[0]]])
            interior_idx = np.setdiff1d(np.arange(len(proj_v)), hull_idx)

        proj_s = _project(samples, polygon) if len(samples) else np.empty((0, 2))
        if len(proj_s):
            ax.scatter(proj_s[:, 0], proj_s[:, 1], s=20, facecolor=main_color,
                       edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)

        if len(hull_pts):
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], facecolor=fill_color, edgecolor='none', zorder=2)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=main_color, linewidth=1.6, zorder=4)
        if len(hull_idx):
            ax.scatter(proj_v[hull_idx, 0], proj_v[hull_idx, 1], s=50,
                       facecolor=main_color, edgecolor="black", linewidth=0.85, zorder=6)

        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)

    sampled = (
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_A,
               markeredgecolor='none', markersize=6, alpha=0.75),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_B,
               markeredgecolor='none', markersize=6, alpha=0.75),
    )
    hull_h = (Line2D([0], [0], color=color_A, linewidth=1.6),
              Line2D([0], [0], color=color_B, linewidth=1.6))
    corner_h = (
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_A,
               markeredgecolor='black', markeredgewidth=0.85, markersize=8),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_B,
               markeredgecolor='black', markeredgewidth=0.85, markersize=8),
    )
    fig.legend(
        handles=[sampled, hull_h, corner_h],
        labels=["Feasible points", "Projected vertex hull",
                "Projected hull corner"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, frameon=False, fontsize=9,
        columnspacing=2.2, handlelength=2.4, handletextpad=0.7,
    )

    plt.savefig(os.path.join(out_dir, f"{out_stub}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{out_stub}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    print("FLARE LP polytope + 5 at.% grid")
    print(f"  Grid step : {int(GRID_STEP * 100)} at.%")
    print(f"  Factor    : {FACTOR}x")

    print("\nBuilding constraint matrix A...")
    A = build_constraint_matrix()

    print("Getting reference vectors...")
    b_W, b_E = get_reference_vectors()

    all_vertices = []
    all_samples = []

    for ref_name, b_ref in [('W', b_W), ('EUROFER', b_E)]:
        print(f"\n--- {FACTOR}x {ref_name} ---")

        G_flare, h_flare = _build_flare_constraint_rows(A, b_ref)
        G, h = build_full_inequality_system(
            A_flare=G_flare,
            b_flare=h_flare,
            lower_bounds=np.zeros(N),
            upper_bounds=np.ones(N),
        )

        t0 = time.time()
        vertices = enumerate_vertices_exact(G, h)
        print(f"  Exact vertices      : {len(vertices)}  |  Time: {time.time() - t0:.2f}s")
        all_vertices.append(vertices)

        t0 = time.time()
        samples = sample_polytope_grid_5atpct_all_multicomponent(G, h)
        print(f"  5 at.% compositions : {len(samples):,}  |  Time: {time.time() - t0:.2f}s")
        all_samples.append(samples)

    vertices_W, vertices_E = all_vertices
    samples_W, samples_E = all_samples

    print("\nGenerating combined figure...")
    plot_feasible_spaces_alloys_AB(
        vertices_A=vertices_W,
        samples_A=samples_W,
        vertices_B=vertices_E,
        samples_B=samples_E,
        out_dir=OUT_DIR,
        out_stub=f"LP-FLARE-polytope-5at-{FACTOR}x"
    )

    print("\nEvaluating responses and saving CSVs...")
    df_W = pd.DataFrame(samples_W, columns=ELEMENTS)
    df_W.insert(0, 'Category', f'{FACTOR}xW')
    df_W = pd.concat([df_W, evaluate_responses(samples_W)], axis=1)
    df_W.to_csv(os.path.join(OUT_DIR, f'LP-FLARE-survivors-5at-{FACTOR}xW.csv'), index=False)

    df_E = pd.DataFrame(samples_E, columns=ELEMENTS)
    df_E.insert(0, 'Category', f'{FACTOR}xEUROFER')
    df_E = pd.concat([df_E, evaluate_responses(samples_E)], axis=1)
    df_E.to_csv(os.path.join(OUT_DIR, f'LP-FLARE-survivors-5at-{FACTOR}xEUROFER.csv'), index=False)

    df_combined = pd.concat([df_W, df_E], ignore_index=True)
    df_combined.to_csv(os.path.join(OUT_DIR, f'LP-FLARE-survivors-5at-{FACTOR}x.csv'), index=False)

    print(f"  LP-FLARE-survivors-5at-{FACTOR}xW.csv       : {len(df_W):,} compositions")
    print(f"  LP-FLARE-survivors-5at-{FACTOR}xEUROFER.csv : {len(df_E):,} compositions")
    print(f"  LP-FLARE-survivors-5at-{FACTOR}x.csv        : {len(df_combined):,} compositions")

    return df_W, df_E, df_combined, vertices_W, vertices_E


if __name__ == "__main__":
    df_W, df_E, df_combined, vertices_W, vertices_E = main()
