#!/usr/bin/env python3
"""
CMS Disappearing track analysis (yaml--embaked efficiency comparison)
For each mass point (m_gluino, m_LSP):
  relative_discrepancy(SR) = (our_eff - cms_eff) / max(cms_eff, floor)
Then average across SRs to a single value per mass point.
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

# ---- Parsing helpers ----

SR_RE = re.compile(r"^SR([1-9]|[1-4]\d)$")  # SR1..SR49
TOP_RE = re.compile(
    r"""^\s*\(\s*(?P<mg>-?\d+)\s*,\s*(?P<mlsp>-?\d+)\s*\)\s*:\s*(?P<body>\{.*\})\s*$""",
    re.DOTALL
)

def parse_embaked_text(text: str):
    """Parse one .embaked file text into (mg, mlsp, payload_dict)."""
    text = text.strip()
    # Case A: "(mg, mlsp): { ... }"
    m = TOP_RE.match(text)
    if m:
        mg = int(m.group("mg"))
        mlsp = int(m.group("mlsp"))
        body = ast.literal_eval(m.group("body"))
        return mg, mlsp, dict(body)
    # Case B: maybe the file is "{(mg, mlsp): {...}}"
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
    """Return long-form CMS df with only SR1..SR49: [m_gluino, m_LSP, SR, cms_eff]."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    dep = data["dependent_variables"]

    # First two are masses
    mglu = [to_int(v.get("value", np.nan)) for v in dep[0]["values"]]
    mlsp = [to_int(v.get("value", np.nan)) for v in dep[1]["values"]]

    rows = []
    for block in dep[2:]:
        name = str(block["header"]["name"])
        if not SR_RE.match(name):
            continue  # skip cross section, all-SRs, baselines, etc.
        vals = [to_float(v.get("value", np.nan)) for v in block["values"]]
        for g, l, e in zip(mglu, mlsp, vals):
            rows.append({"m_gluino": g, "m_LSP": l, "SR": name, "cms_eff": e})
    return pd.DataFrame(rows, columns=["m_gluino","m_LSP","SR","cms_eff"])

def load_ours_long(embaked_dir: str) -> pd.DataFrame:
    """Return long-form OUR df with only SR1..SR49: [m_gluino, m_LSP, SR, our_eff]."""
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

# ---- Computation ----

def compute_relative_discrepancy(merged: pd.DataFrame, floor: float = 1e-5) -> pd.DataFrame:
    """Compute relative discrepancy per row (SR) and then aggregate by mass point."""
    cms = merged["cms_eff"].to_numpy(dtype=float)
    ours = merged["our_eff"].to_numpy(dtype=float)
    denom = np.maximum(cms, floor)
    rb = (ours - cms) / denom
    out = merged.copy()
    out["rel_discrepancy"] = rb
    return out

def aggregate_discrepancy_per_mass(rb_long: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Aggregate SR-level rel_discrepancy to one value per (m_gluino, m_LSP)."""
    if how == "median":
        agg = rb_long.groupby(["m_gluino","m_LSP"])["rel_discrepancy"].median().reset_index(name="avg_rel_discrepancy")
    else:
        agg = rb_long.groupby(["m_gluino","m_LSP"])["rel_discrepancy"].mean().reset_index(name="avg_rel_discrepancy")
    return agg.sort_values(["m_gluino","m_LSP"]).reset_index(drop=True)

# ---- Plotting ----

def plot_reldiscrepancy_heatmap(pivot: pd.DataFrame, out_png: str = "reldiscrepancy_heatmap.png",
                         title: str = "T5btbt average relative discrepancy",
                         fmt: str = ".2f", fontsize: int = 10,
                         figwidth: float = 13, figheight: float = 9,
                         tick_every_x: int = 3, tick_every_y: int = 2,
                         rotate_xticks: int = 45, vmax_user: float = None):
    """Draw heatmap with centered annotations; NaNs are gray (no 'NaN' text)."""
    data = pivot.values
    nrows, ncols = data.shape
    xmin, xmax = float(pivot.columns.min()), float(pivot.columns.max())
    ymin, ymax = float(pivot.index.min()), float(pivot.index.max())
    dx = (xmax - xmin) / ncols if ncols else 1.0
    dy = (ymax - ymin) / nrows if nrows else 1.0

    # Symmetric color scale around 0
    if vmax_user is None:
        # robust symmetric limit (98th percentile of |data|)
        finite_vals = np.abs(data[np.isfinite(data)])
        vmax = np.percentile(finite_vals, 98) if finite_vals.size else 1.0
        vmax = max(vmax, 1e-3)
    else:
        vmax = float(vmax_user)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    im = ax.imshow(masked, origin="lower", aspect="auto",
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=-vmax, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average relative discrepancy across SRs")

    # ticks at centers with labels
    xs = [xmin + (j + 0.5) * dx for j in range(ncols)]
    ys = [ymin + (i + 0.5) * dy for i in range(nrows)]
    ax.set_xticks(xs[::max(1, tick_every_x)])
    ax.set_yticks(ys[::max(1, tick_every_y)])
    ax.set_xticklabels(list(pivot.columns)[::max(1, tick_every_x)], rotation=rotate_xticks, ha="right")
    ax.set_yticklabels(list(pivot.index)[::max(1, tick_every_y)])

    # annotations at centers (black text)
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            cx = xmin + (j + 0.5) * dx
            cy = ymin + (i + 0.5) * dy
            ax.text(cx, cy, format(val, fmt), ha="center", va="center", fontsize=fontsize, color="black")

    ax.set_xlabel("m_gluino (GeV)")
    ax.set_ylabel("m_LSP (GeV)")
    ax.set_title(title)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved {out_png}")

# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description="Average relative discrepancy heatmap (SR1..SR49 only).")
    ap.add_argument("--yaml", required=True, help="Path to CMS YAML")
    ap.add_argument("--embaked", required=True, help="Folder with .embaked files")
    ap.add_argument("--floor", type=float, default=1e-5, help="Denominator floor for CMS efficiency")
    ap.add_argument("--agg", choices=["mean","median"], default="mean", help="Aggregate across SRs")
    ap.add_argument("--fmt", type=str, default=".2f", help="Annotation number format")
    ap.add_argument("--fontsize", type=int, default=10, help="Annotation font size")
    ap.add_argument("--figwidth", type=float, default=13)
    ap.add_argument("--figheight", type=float, default=9)
    ap.add_argument("--tick-every-x", type=int, default=3)
    ap.add_argument("--tick-every-y", type=int, default=2)
    ap.add_argument("--rotate-xticks", type=int, default=45)
    ap.add_argument("--vmax", type=float, default=None, help="Symmetric color scale max (if None, auto)")
    args = ap.parse_args()

    cms_long = load_cms_long(args.yaml)
    ours_long = load_ours_long(args.embaked)
    merged = pd.merge(cms_long, ours_long, on=["m_gluino","m_LSP","SR"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between CMS YAML and .embaked on (m_gluino, m_LSP, SR).")

    rb_long = compute_relative_discrepancy(merged, floor=args.floor)
    agg_df = aggregate_discrepancy_per_mass(rb_long, how=args.agg)

    pivot = agg_df.pivot(index="m_LSP", columns="m_gluino", values="avg_rel_discrepancy")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    plot_reldiscrepancy_heatmap(
        pivot,
        out_png="reldiscrepancy_heatmap.png",
        title="T5btbt average relative discrepancy",
        fmt=args.fmt, fontsize=args.fontsize,
        figwidth=args.figwidth, figheight=args.figheight,
        tick_every_x=args.tick_every_x, tick_every_y=args.tick_every_y,
        rotate_xticks=args.rotate_xticks, vmax_user=args.vmax
    )

if __name__ == "__main__":
    main()
