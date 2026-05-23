import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# STYLE  (matches LP_FLARE_GRID.py)
# =========================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times", "Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.9,
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
})

# =========================
# FILES
# =========================
LP_FILE    = "LP-FLARE-survivors-5at-10x.csv"
BRUTE_FILE = "FLARE-survivors-5at-10x.csv"
OUT_STUB   = "LP-FLARE-parity-5at-10x"

ELEMENTS = ["Ti", "V", "Ta", "Nb", "Mo", "Zr", "Cr", "Hf", "Fe", "Re", "W"]

# (short name, axis-label with units, brute column, LP column)
PROPERTIES = [
    ("M_GAS", r"$M_\mathrm{GAS}$ (g/g)", "M_GAS_brute", "M_GAS_lp"),
    ("HT_T",  r"$H_\mathrm{T}$ (W/g)",   "HT_T_brute",  "HT_T_lp"),
    ("DPA",   r"DPA",                    "DPA_brute",   "DPA_lp"),
    ("DOSE",  r"DOSE (Sv/h)",            "DOSE",        "DOSE_1s"),
]

# =========================
# LOAD, DEDUP, MERGE
# =========================
df_lp    = pd.read_csv(LP_FILE).drop_duplicates(subset=ELEMENTS)
df_brute = pd.read_csv(BRUTE_FILE).drop_duplicates(subset=ELEMENTS)

df = df_brute.merge(df_lp, on=ELEMENTS, suffixes=("_brute", "_lp"))
print(f"Unique compositions compared: {len(df):,}")

# =========================
# PLOT
# =========================
POINT_COLOR = "#1F77B4"

fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
fig.subplots_adjust(top=0.96, bottom=0.08, left=0.10, right=0.97,
                    hspace=0.32, wspace=0.32)

for ax, (short_name, label, brute_col, lp_col) in zip(axes.ravel(), PROPERTIES):
    x = pd.to_numeric(df[brute_col], errors="coerce")
    y = pd.to_numeric(df[lp_col],    errors="coerce")
    valid = x.notna() & y.notna() & (x > 0) & (y > 0)
    x = x[valid].to_numpy()
    y = y[valid].to_numpy()

    # Metrics
    diff   = y - x
    mae    = np.mean(np.abs(diff))
    rmse   = np.sqrt(np.mean(diff ** 2))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"  {short_name:6s}: R2={r2:.10f}  MAE={mae:.2e}  RMSE={rmse:.2e}  N={len(x):,}")

    # Identity line (behind points)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.3,
            color="black", alpha=0.7, label="Ideal (y = x)", zorder=2)

    # Scatter
    ax.scatter(x, y, s=18, alpha=0.7, color=POINT_COLOR,
               edgecolors="black", linewidths=0.4, zorder=3)

    # Log-log axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mn / 1.5, mx * 1.5)
    ax.set_ylim(mn / 1.5, mx * 1.5)

    # Labels and title
    ax.set_xlabel(f"Brute-force  {label}")
    ax.set_ylabel(f"LP  {label}")
    ax.set_title(short_name, fontsize=11, fontweight="bold", pad=8)

    # Metrics inset
    txt = (f"$R^2$ = {r2:.10f}\n"
           f"MAE  = {mae:.2e}\n"
           f"RMSE = {rmse:.2e}\n"
           f"$N$ = {len(x):,}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="gray", linewidth=0.5, alpha=0.9))

    # Legend for the dashed identity line
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Aspect + grid
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)

plt.savefig(f"{OUT_STUB}.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{OUT_STUB}.pdf", bbox_inches="tight")
plt.show()