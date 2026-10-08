"""Regenerate the current paper and appendix diagnostic figures as vector PDFs.

These figures use protocol-matched, checkpoint-specific values reported in the
paper. The crossover curves use the saved seed-42 CQG-Single Top-200 output and
TIGER beam-200 output; every plotted depth is measured rather than interpolated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", format="pdf")
    plt.close(fig)


def figure2():
    labels = [
        "Q1 (rare)\n$n$=193 (3.2%)",
        "Q2\n$n$=581 (9.6%)",
        "Q3\n$n$=1,322 (21.9%)",
        "Q4 (popular)\n$n$=3,944 (65.3%)",
    ]
    methods = [
        ("TIGER", [27.979, 49.914, 59.909, 73.352], "#E69F00", ""),
        ("CQG-Single", [29.534, 51.119, 61.876, 73.352], "#0072B2", "//"),
        ("CQG-AR", [26.425, 49.742, 60.514, 72.769], "#009E73", "\\\\"),
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    width = 0.24
    for method_index, (method, values, color, hatch) in enumerate(methods):
        offset = (method_index - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            label=method,
            hatch=hatch,
            edgecolor="white" if hatch else color,
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="black",
            )
    ax.axhline(66.705, color="#9A6700", lw=2.0, ls="--", zorder=0)
    ax.text(
        0.02,
        68.2,
        "TIGER TargetInOutput@100: 66.7%",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#805500",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.set_ylabel("TargetInOutput@100 (%)", fontsize=14, color="black")
    ax.set_xticks(x, labels, fontsize=10, color="black")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=11, colors="black")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        title="Overall: TIGER 66.7%  |  Single 67.3%  |  AR 66.4%",
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        title_fontsize=9.5,
        ncol=3,
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_reachability.pdf")


def figure3():
    ks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150])
    panels = [
        ("Q1 (rare, 193 users)",
         [0.062176, 0.119171, 0.145078, 0.165803, 0.186528, 0.191710, 0.212435, 0.243523, 0.284974, 0.295337, 0.316062, 0.357513],
         [0.082902, 0.108808, 0.145078, 0.160622, 0.176166, 0.196891, 0.227979, 0.248705, 0.259067, 0.264249, 0.305699, 0.336788],
         [0.067358, 0.103627, 0.134715, 0.165803, 0.191710, 0.202073, 0.222798, 0.259067, 0.264249, 0.279793, 0.305699, 0.316062],
         0.352332),
        ("Q2 (581 users)",
         [0.177281, 0.271945, 0.323580, 0.361446, 0.395869, 0.432014, 0.450947, 0.480207, 0.492255, 0.511188, 0.550775, 0.583477],
         [0.149742, 0.253012, 0.309811, 0.352840, 0.390706, 0.418244, 0.437177, 0.461274, 0.481928, 0.497418, 0.528399, 0.564544],
         [0.175559, 0.254733, 0.313253, 0.352840, 0.387263, 0.423408, 0.442341, 0.464716, 0.487091, 0.499139, 0.524957, 0.561102],
         0.617900),
        ("Q3 (1,322 users)",
         [0.251891, 0.344932, 0.420575, 0.467474, 0.503026, 0.531014, 0.562784, 0.584720, 0.600605, 0.618759, 0.642965, 0.673979],
         [0.239032, 0.343419, 0.407716, 0.448563, 0.484115, 0.520424, 0.546142, 0.568835, 0.583964, 0.605144, 0.633888, 0.656581],
         [0.243570, 0.341150, 0.407716, 0.450832, 0.481089, 0.509077, 0.536309, 0.558245, 0.580938, 0.599092, 0.631619, 0.653555],
         0.695159),
        ("Q4 (popular, 3,944 users)",
         [0.355477, 0.475406, 0.547921, 0.597363, 0.633114, 0.662018, 0.686359, 0.703854, 0.718560, 0.733519, 0.757099, 0.784229],
         [0.349391, 0.463489, 0.531947, 0.580882, 0.617394, 0.648834, 0.674696, 0.697769, 0.712475, 0.727688, 0.749746, 0.779158],
         [0.347363, 0.461207, 0.530680, 0.583418, 0.627282, 0.650862, 0.679513, 0.702333, 0.720842, 0.733519, 0.761917, 0.789554],
         0.827333),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.7), sharex=False)
    blue, green, gold, limit_color = "#0072B2", "#009E73", "#E69F00", "#D55E00"
    for idx, (ax, (title, single, ar, tiger, limit)) in enumerate(zip(axes.flat, panels)):
        cqg_arr = np.asarray(single)
        tiger_arr = np.asarray(tiger)
        y_max = min(1.0, max(max(single), max(ar), max(tiger), limit) * 1.08)
        ax.fill_between(
            [ks.min() - 5, ks.max() + 5],
            limit,
            y_max,
            color=limit_color,
            alpha=0.05,
            zorder=0,
        )
        ax.fill_between(
            ks,
            tiger_arr,
            cqg_arr,
            where=cqg_arr >= tiger_arr,
            interpolate=True,
            color=blue,
            alpha=0.14,
            linewidth=0,
            label="CQG-Single advantage",
        )
        ax.plot(ks, single, "-o", color=blue, lw=1.15, ms=3.2, label="CQG-Single")
        ax.plot(ks, ar, "-^", color=green, lw=1.15, ms=3.3, label="CQG-AR")
        ax.plot(ks, tiger, "-s", color=gold, lw=1.15, ms=3.2, label="TIGER")
        ax.axhline(
            limit,
            color=limit_color,
            lw=1.1,
            ls=(0, (5, 3)),
            alpha=0.65,
            label="TIGER reachability limit ($B=200$)",
        )
        limit_label = ax.text(
            11,
            limit + y_max * 0.012,
            f"Reachability limit ({limit:.1%})",
            ha="left",
            va="bottom",
            fontsize=7.5,
            fontstyle="italic",
            color=limit_color,
        )
        limit_label.set_path_effects(
            [pe.withStroke(linewidth=2.8, foreground="white")]
        )
        if idx == 0:
            ax.annotate(
                "CQG-Single = TIGER\n$K=40$, Recall=0.1658",
                xy=(40, single[3]),
                xytext=(63, 0.105),
                fontsize=7.5,
                color="#444444",
                ha="center",
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#666666",
                    "connectionstyle": "arc3,rad=-0.18",
                    "linewidth": 0.9,
                },
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "linewidth": 0.6,
                    "alpha": 0.95,
                },
                zorder=10,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([10, 50, 100, 150])
        if idx in (2, 3):
            ax.set_xlabel("Retrieval depth $K$")
        ax.set_xlim(5, 155)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)
        if idx in (0, 2):
            ax.set_ylabel("Recall")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_crossover.pdf")


def figure4():
    labels = [
        "v3\nDual",
        "v4\nDual",
        "v5h\nMSE-warm",
        "v6\nQuery-only",
        "v7\nText-aware",
        "Scratch MSE\n(LLM diagnostic)",
    ]
    values = [0.003, 0.0, 0.0, 0.044, 0.212, 0.248]
    colors = ["#D95F02", "#8C8C8C", "#8C8C8C", "#984807", "#984807", "#E69F00"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(x, values, width=0.64, color=colors, edgecolor="#3B3B3B", linewidth=1.2)
    ax.axhline(0.248, color="#3B94C5", lw=2, ls="--")
    ax.text(3.75, 0.272, "Scratch diagnostic baseline", color="#0072B2",
            ha="center", fontsize=10)
    for index in (3, 4, 5):
        ax.text(x[index], values[index] + 0.006, f"{values[index]:.3f}", ha="center", fontsize=11)
    for index, text in enumerate(["I-I=0.93", "I-I=0.93", "I-I=0.994"]):
        ax.text(
            x[index],
            0.119,
            text,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="#E67E22", ec="#333333", lw=1.3),
        )
    ax.text(1.0, 0.018, "Item embedding\ncollapse", color="#D95F02", ha="center", fontstyle="italic")
    ax.text(3.25, 0.072, "Loss-metric\ndecoupling", color="#8C3F0A", ha="center", fontstyle="italic")
    ax.set_ylabel("Test R@10", fontsize=13)
    ax.set_xticks(x, labels)
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_rotation(28)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_y(-0.015 if index % 2 == 0 else -0.045)
    ax.tick_params(axis="x", labelsize=10, pad=5)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 0.28)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.31, left=0.12, right=0.98, top=0.96)
    save(fig, "fig4_llm_diagnostic.pdf")


def figure5():
    """Consolidate the objective, visibility, and temperature diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    blue, green, gold, gray = "#0072B2", "#009E73", "#E69F00", "#777777"

    # (a) Frozen-item objective landscape.
    ax = axes[0]
    labels = ["Single\nCE", "Single\nMSE", "Single\nInfoNCE",
              "AR\nCE", "AR\nMSE", "AR\nInfoNCE"]
    values = [0.3114, 0.2425, 0.2386, 0.2982, 0.2424, 0.2324]
    colors = [blue, gray, gray, green, gray, gray]
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.72)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.2797, color=gold, ls="--", lw=1.3, label="SASRec BPR")
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.5)
    ax.set_ylabel("Test R@10")
    ax.set_title("(a) Training objective")
    ax.set_ylim(0.20, 0.33)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # (b) Full-catalog versus sampled CE, mean +/- sample SD.
    ax = axes[1]
    vis_labels = ["Full-catalog\nCE", "Sampled CE\n(256 neg.)"]
    vis_values = [0.2982, 0.2895]
    vis_sd = [0.0042, 0.0077]
    bars = ax.bar([0, 1], vis_values, yerr=vis_sd, capsize=4,
                  color=[green, "#56B4E9"], width=0.62)
    for bar, value in zip(bars, vis_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.011,
                f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1], vis_labels, fontsize=8)
    ax.set_title("(b) Catalog visibility")
    ax.set_ylim(0.26, 0.325)
    ax.text(
        0.5,
        0.319,
        r"$\Delta$R@10 = -0.0087",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
              "pad": 1.2},
    )

    # (c) Temperature sensitivity.
    ax = axes[2]
    temperatures = np.array([0.07, 0.10, 0.20])
    single = [0.3114, 0.3070, 0.2846]
    ar = [0.2972, 0.2886, 0.2725]
    ax.plot(temperatures, single, "-o", color=blue, label="CQG-Single")
    ax.plot(temperatures, ar, "-s", color=green, label="CQG-AR")
    ax.axhline(0.2425, color=gray, ls="--", lw=1.2,
               label="Single MSE")
    for xs, ys, color in [(temperatures, single, blue),
                          (temperatures, ar, green)]:
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.003, f"{y:.3f}", ha="center",
                    va="bottom", fontsize=7.5, color=color)
    ax.set_xticks(temperatures, ["0.07", "0.10", "0.20"])
    ax.set_xlabel(r"Temperature $\tau$")
    ax.set_title("(c) Temperature")
    ax.set_ylim(0.23, 0.325)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig5_ce_mse.pdf")


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    figure5()
