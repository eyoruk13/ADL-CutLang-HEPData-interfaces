#!/usr/bin/env python3
"""
CMS Disappearing track analysis (yaml--embaked efficiency comparison)
Per-mass-point correlation heatmap (ONLY SR1..SR49), robust parsing,
with centered annotations and improved labels/title.

Title fixed to: "T5btbt per mass point correlation"
X label: "m_gluino (GeV)"
Y label: "m_LSP (GeV)"

Author: Ekin Sıla Yörük
"""
import os
import re
import ast
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

SR_RE = re.compile(r"^SR([1-9]|[1-4]\d)$")  # SR1..SR49
TOP_RE = re.compile(
    r"""^\s*\(\s*(?P<mg>-?\d+)\s*,\s*(?P<mlsp>-?\d+)\s*\)\s*:\s*(?P<body>\{.*\})\s*$""",
    re.DOTALL
)

def parse_embaked_text(text: str):
    text = text.strip()
    m = TOP_RE.match(text)
    if m:
        mg = int(m.group("mg"))
        mlsp = int(m.group("mlsp"))
        body = ast.literal_eval(m.group("body"))
        return mg, mlsp, dict(body)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict) and len(obj) == 1:
            (mg, mlsp), payload = next(iter(obj.items()))
            return int(mg), int(mlsp), dict(payload)
    except Exception:
        pass
    raise ValueError("Invalid .embaked format; expected '(mg, mlsp): { ... }' or '{(mg, mlsp): {...}}'.")

def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return int(v)

def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def load_cms_long(yaml_path: str) -> pd.DataFrame:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    dep = data["dependent_variables"]
    mglu = [to_int(v.get("value", np.nan)) for v in dep[0]["values"]]
    mlsp = [to_int(v.get("value", np.nan)) for v in dep[1]["values"]]
    rows = []
    for block in dep[2:]:
        name = str(block["header"]["name"])
        if not SR_RE.match(name):
            continue
        vals = [to_float(v.get("value", np.nan)) for v in block["values"]]
        for g, l, e in zip(mglu, mlsp, vals):
            rows.append({"m_gluino": g, "m_LSP": l, "SR": name, "cms_eff": e})
    return pd.DataFrame(rows, columns=["m_gluino","m_LSP","SR","cms_eff"])

def load_ours_long(embaked_dir: str) -> pd.DataFrame:
    rows = []
    for fname in os.listdir(embaked_dir):
        if not fname.endswith(".embaked"):
            continue
        path = os.path.join(embaked_dir, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            mg, mlsp, payload = parse_embaked_text(text)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
            continue
        for k, v in payload.items():
            k = str(k)
            if SR_RE.match(k):
                rows.append({"m_gluino": int(mg), "m_LSP": int(mlsp), "SR": k, "our_eff": to_float(v)})
    return pd.DataFrame(rows, columns=["m_gluino","m_LSP","SR","our_eff"])

def compute_corr_per_mass(merged: pd.DataFrame, method: str = "pearson",
                          zero_thr: float = 1e-6, min_srs: int = 1,
                          treat_both_const_equal: bool = True,
                          treat_both_near_zero_equal: bool = True) -> pd.DataFrame:
    out = []
    for (g, l), gdf in merged.groupby(["m_gluino", "m_LSP"]):
        x = gdf["cms_eff"].to_numpy(dtype=float)
        y = gdf["our_eff"].to_numpy(dtype=float)
        corr = np.nan
        n = len(x)
        if n >= min_srs:
            varx = np.max(np.abs(x - x.mean())) if n else 0.0
            vary = np.max(np.abs(y - y.mean())) if n else 0.0
            both_const = (varx <= zero_thr) and (vary <= zero_thr)
            both_near_zero = (np.max(np.abs(x)) <= zero_thr) and (np.max(np.abs(y)) <= zero_thr)
            equal_within_thr = np.max(np.abs(x - y)) <= zero_thr
            if treat_both_const_equal and both_const and equal_within_thr:
                corr = 1.0
            elif treat_both_near_zero_equal and both_near_zero:
                corr = 1.0
            elif varx <= zero_thr or vary <= zero_thr:
                corr = np.nan
            else:
                if method == "spearman":
                    corr = spearmanr(x, y).correlation
                else:
                    corr = pearsonr(x, y)[0]
        out.append({"m_gluino": g, "m_LSP": l, "corr": corr, "nSR": n})
    return pd.DataFrame(out).sort_values(["m_gluino","m_LSP"]).reset_index(drop=True)

def plot_heatmap_annot(pivot: pd.DataFrame, fmt: str = ".2f", fontsize: int = 10, dpi: int = 160,
                       figwidth: float = 13, figheight: float = 9,
                       tick_every_x: int = 3, tick_every_y: int = 2,
                       annot_skip_x: int = 1, annot_skip_y: int = 1,
                       rotate_xticks: int = 45, show_nan: bool = False, nan_text: str = "NA",
                       save_png: str = "corr_heatmap_v10b.png", save_pdf: bool = False):
    data = pivot.values
    nrows, ncols = data.shape
    xmin, xmax = float(pivot.columns.min()), float(pivot.columns.max())
    ymin, ymax = float(pivot.index.min()), float(pivot.index.max())
    dx = (xmax - xmin) / ncols if ncols else 1.0
    dy = (ymax - ymin) / nrows if nrows else 1.0

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=(figwidth, figheight), dpi=dpi)
    im = ax.imshow(masked, origin="lower", aspect="auto",
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=-1, vmax=1, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation across SRs")

    # Ticks at centers
    xs = [xmin + (j + 0.5) * dx for j in range(ncols)]
    ys = [ymin + (i + 0.5) * dy for i in range(nrows)]
    ax.set_xticks(xs[::max(1, tick_every_x)])
    ax.set_yticks(ys[::max(1, tick_every_y)])
    ax.set_xticklabels(list(pivot.columns)[::max(1, tick_every_x)], rotation=rotate_xticks, ha="right")
    ax.set_yticklabels(list(pivot.index)[::max(1, tick_every_y)])

    # Annotations
    for i in range(nrows):
        for j in range(ncols):
            if (j % max(1, annot_skip_x)) or (i % max(1, annot_skip_y)):
                continue
            val = data[i, j]
            cx = xmin + (j + 0.5) * dx
            cy = ymin + (i + 0.5) * dy
            if np.isnan(val):
                if show_nan:
                    ax.text(cx, cy, nan_text, ha="center", va="center", fontsize=fontsize, color="black")
                continue
            ax.text(cx, cy, format(val, fmt), ha="center", va="center", fontsize=fontsize, color="black")

    # Labels & title
    ax.set_xlabel("m_gluino (GeV)", fontsize=12)
    ax.set_ylabel("m_LSP (GeV)", fontsize=12)
    ax.set_title("T5btbt per mass point correlation", fontsize=14)

    # Extra bottom margin to ensure xlabel is visible even with rotated xticks
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(save_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    print(f"[DONE] Saved {save_png}")

def main():
    ap = argparse.ArgumentParser(description="T5btbt per mass point correlation heatmap (SR1..SR49 only).")
    ap.add_argument("--yaml", required=True, help="Path to CMS YAML")
    ap.add_argument("--embaked", required=True, help="Folder with .embaked files")
    ap.add_argument("--method", choices=["pearson","spearman"], default="pearson")
    ap.add_argument("--zero-threshold", type=float, default=1e-6)
    ap.add_argument("--min-srs", type=int, default=1)
    ap.add_argument("--fmt", type=str, default=".2f")
    ap.add_argument("--fontsize", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--figwidth", type=float, default=13)
    ap.add_argument("--figheight", type=float, default=9)
    ap.add_argument("--tick-every-x", type=int, default=3)
    ap.add_argument("--tick-every-y", type=int, default=2)
    ap.add_argument("--annot-skip-x", type=int, default=1)
    ap.add_argument("--annot-skip-y", type=int, default=1)
    ap.add_argument("--rotate-xticks", type=int, default=45)
    ap.add_argument("--show-nan", action="store_true")
    ap.add_argument("--nan-text", type=str, default="NA")
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    cms_long = load_cms_long(args.yaml)
    ours_long = load_ours_long(args.embaked)
    merged = pd.merge(cms_long, ours_long, on=["m_gluino","m_LSP","SR"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between CMS YAML and .embaked on (m_gluino, m_LSP, SR).")

    # Correlation per mass point
    from scipy.stats import spearmanr  # lazy import for completeness
    corr_df = compute_corr_per_mass(
        merged, method=args.method, zero_thr=args.zero_threshold, min_srs=args.min_srs,
        treat_both_const_equal=True, treat_both_near_zero_equal=True
    )

    # Pivot and order
    piv = corr_df.pivot(index="m_LSP", columns="m_gluino", values="corr")
    piv = piv.sort_index(axis=0).sort_index(axis=1)

    plot_heatmap_annot(
        piv, fmt=args.fmt, fontsize=args.fontsize, dpi=args.dpi,
        figwidth=args.figwidth, figheight=args.figheight,
        tick_every_x=args.tick_every_x, tick_every_y=args.tick_every_y,
        annot_skip_x=args.annot_skip_x, annot_skip_y=args.annot_skip_y,
        rotate_xticks=args.rotate_xticks,
        show_nan=args.show_nan, nan_text=args.nan_text,
        save_png="T5btbt_per_masspoint_correlation.png",
        save_pdf=args.save_pdf
    )

if __name__ == "__main__":
    main()
