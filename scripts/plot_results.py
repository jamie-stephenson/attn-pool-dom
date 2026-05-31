"""Plot CAA dose-response curves and refusal bar charts from results/ JSON.

Run on the box (matplotlib is a remote dep); PNGs land in results/ and sync back
to the Mac via git. `uv run python scripts/plot_results.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _coef_items(curve: dict) -> list[tuple[float, float]]:
    return sorted(((float(k), float(v)) for k, v in curve.items()), key=lambda x: x[0])


def plot_caa(path: Path, out: Path) -> None:
    d = json.loads(path.read_text())
    res = d["results"]
    plt.figure(figsize=(7, 5))
    for spec, info in res.items():
        xs_ys = _coef_items(info["curve"])
        xs = [x for x, _ in xs_ys]
        ys = [y for _, y in xs_ys]
        plt.plot(xs, ys, marker="o", label=f"{spec} (|v|={info['norm']:.2f})")
    plt.axhline(0.5, ls=":", c="gray", lw=1)
    plt.xlabel("steering coefficient")
    plt.ylabel("P(behaviour-matching answer)")
    plt.title(f"CAA sycophancy dose-response (layer {d.get('layer')})")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


def plot_refusal(path: Path, out: Path) -> None:
    d = json.loads(path.read_text())
    base = d.get("baseline", {})
    specs = [k for k in d if k != "baseline"]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ablation: harmful refusal, baseline vs each spec
    ax1.axhline(base.get("harmful_refusal", 0), ls="--", c="k", label="no-steer harmful")
    ax1.bar(range(len(specs)), [d[s]["ablate_harmful_refusal"] for s in specs])
    ax1.set_xticks(range(len(specs)))
    ax1.set_xticklabels(specs, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("refusal rate on HARMFUL (ablated)")
    ax1.set_title("Ablation = bypass (lower is stronger)")
    ax1.legend(fontsize=7)

    # addition: harmless refusal vs coef per spec
    for s in specs:
        xs_ys = _coef_items(d[s]["add_harmless_refusal"])
        ax2.plot([x for x, _ in xs_ys], [y for _, y in xs_ys], marker="o", label=s)
    ax2.axhline(base.get("harmless_refusal", 0), ls="--", c="k", label="no-steer harmless")
    ax2.set_xlabel("addition coefficient")
    ax2.set_ylabel("refusal rate on HARMLESS (induced)")
    ax2.set_title("Addition = induce (higher is stronger)")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print("wrote", out)


def main() -> None:
    root = Path("results")
    for p in sorted((root / "caa").glob("*.json")) if (root / "caa").exists() else []:
        plot_caa(p, p.with_suffix(".png"))
    for p in sorted((root / "refusal").glob("*.json")) if (root / "refusal").exists() else []:
        plot_refusal(p, p.with_suffix(".png"))


if __name__ == "__main__":
    main()
