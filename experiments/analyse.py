"""
All numbers and figures for the A3 report, from results/<tag>/runs.csv and the
traces. Only runs with lambda_3 > 0 are analysed (the training set determines the
linear representation; with lambda_3 = 0 no run reached RQI > 0.95 in any setting).

Outputs
  results/analysis/summary.txt   every number quoted in the report
  results/analysis/*.csv         per-setting tables
  figures/fig_*.pdf (+ .png)     report figures

Statistics. Runs with the same seed share targets and initialisation and have
nested training sets, so they are not independent across training sizes. Settings
are therefore compared with an exact sign test over the 20 seeds (seed = unit),
as in A2. Everything else is descriptive.

Usage: python experiments/analyse.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "analysis")
FIG = os.path.join(ROOT, "figures")

# setting tag -> label used in the report
SETTINGS = {
    "dec_lr_1e-3": "decoder lr $10^{-3}$",
    "released": "released",
    "emb_lr_3e-3": "embedding lr $3{\\cdot}10^{-3}$",
    "emb_lr_1e-2": "embedding lr $10^{-2}$",
    "dec_lr_1e-5": "decoder lr $10^{-5}$",
}
PLAIN = {"dec_lr_1e-3": "decoder lr 1e-3", "released": "released (emb 1e-3, dec 1e-4)",
         "emb_lr_3e-3": "embedding lr 3e-3", "emb_lr_1e-2": "embedding lr 1e-2",
         "dec_lr_1e-5": "decoder lr 1e-5"}

# colours: validated categorical slots 1-2 (blue, orange); ink for text and axes
SUCCESS, FAILURE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "savefig.dpi": 300,
    "pdf.fonttype": 42, "font.family": "serif",
})
COL = 3.4   # one ISBI column, inches
FULL = 7.0  # full page width


def load(tag):
    rows = list(csv.DictReader(open(os.path.join(RES, tag, "runs.csv"))))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
        r["ok"] = r["reached_rqi"] == 1
        r["fit"] = r["reached_train90"] == 1
    return rows


def trace(tag, r):
    return np.load(os.path.join(RES, tag, "traces",
                                f"n{int(r['n_train']):02d}_s{int(r['seed']):02d}.npz"))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def per_seed_successes(rows):
    c = np.zeros(20, int)
    for r in rows:
        c[int(r["seed"])] += r["ok"]
    return c


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    allrows = {t: load(t) for t in SETTINGS}
    data = {t: [r for r in rows if r["lambda3"] > 1e-9] for t, rows in allrows.items()}
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    n_pos = len(data["released"])
    say(f"Runs per setting: 220 (11 sizes x 20 seeds); lambda_3 > 0: {n_pos}")
    for t, rows in allrows.items():
        z = [r for r in rows if r["lambda3"] <= 1e-9]
        say(f"  {PLAIN[t]}: lambda_3 = 0 runs reaching RQI > 0.95: {sum(r['ok'] for r in z)}/{len(z)}")

    # ---------------- 1. settings vs released (H3 and decoder lr) ----------------
    say("\n1. Successes (RQI > 0.95 within 10^4 steps) and seed-level sign test vs released")
    c0 = per_seed_successes(data["released"])
    tab = []
    for t, rows in data.items():
        c = per_seed_successes(rows)
        b, w = int((c > c0).sum()), int((c < c0).sum())
        p = binomtest(b, b + w).pvalue if (t != "released" and b + w) else np.nan
        tab.append(dict(setting=PLAIN[t], successes=int(c.sum()), runs=len(rows),
                        seeds_better=b, seeds_worse=w, seeds_tied=20 - b - w, sign_test_p=p))
        say(f"  {PLAIN[t]:32s} {c.sum():3d}/{len(rows)}  seeds better/worse/tied "
            f"{b}/{w}/{20 - b - w}  p = {p:.2g}")
    c1, c2 = per_seed_successes(data["emb_lr_3e-3"]), per_seed_successes(data["emb_lr_1e-2"])
    b, w = int((c2 > c1).sum()), int((c2 < c1).sum())
    say(f"  embedding lr 1e-2 vs 3e-3: seeds better/worse {b}/{w}, p = {binomtest(b, b + w).pvalue:.2g}")
    write_csv("settings.csv", tab)

    # ---------------- 2. H2: RQI at the decoder fit ----------------
    say("\n2. RQI at the step of 90% training accuracy ('the fit'), lambda_3 > 0 runs")
    tab = []
    for t, rows in data.items():
        fit = [r for r in rows if r["fit"]]
        S = [r["rqi_at_train90"] for r in fit if r["ok"]]
        F = [r["rqi_at_train90"] for r in fit if not r["ok"]]
        S100 = [r["rqi_at_train100"] for r in rows if r["reached_train100"] == 1 and r["ok"]]
        F100 = [r["rqi_at_train100"] for r in rows if r["reached_train100"] == 1 and not r["ok"]]
        succ = [r for r in rows if r["ok"]]
        late = [r for r in succ if r["fit"] and r["iter_rqi"] > r["iter_train90"]]
        nofit = [r for r in rows if not r["fit"]]
        lead = [r["iter_train90"] - r["iter_rqi"] for r in succ if r["fit"]]
        fail_change = [r["final_rqi"] - r["rqi_at_train90"] for r in fit if not r["ok"]]
        row = dict(setting=PLAIN[t], reached_fit=len(fit), successes_with_fit=len(S),
                   failures_with_fit=len(F),
                   median_rqi_fit_success=med(S), median_rqi_fit_failure=med(F),
                   max_rqi_fit_failure=max(F) if F else np.nan,
                   min_rqi_fit_success=min(S) if S else np.nan,
                   auc_90=auc(S, F), auc_100=auc(S100, F100),
                   successes_formed_after_fit=len(late),
                   median_lead_steps=med(lead),
                   median_rqi_change_after_fit_failures=med(fail_change),
                   no_fit_in_budget=len(nofit),
                   no_fit_but_success=sum(r["ok"] for r in nofit))
        tab.append(row)
        say(f"  {PLAIN[t]}: fit reached {len(fit)}/{len(rows)}; median RQI at fit success "
            f"{row['median_rqi_fit_success']:.2f} / failure {row['median_rqi_fit_failure']:.2f}; "
            f"AUC {row['auc_90']:.3f} (100% threshold: {row['auc_100']:.3f}); "
            f"successes formed after fit {len(late)}/{len(succ)}; "
            f"median lead {row['median_lead_steps']:.0f} steps; failures: median RQI change "
            f"after fit {row['median_rqi_change_after_fit_failures']:+.3f}; "
            f"no fit in 10^4 steps {len(nofit)} (of which success {row['no_fit_but_success']})")
        for r in late:
            say(f"      formed after fit: n={int(r['n_train'])} seed={int(r['seed'])}, "
                f"RQI at fit {r['rqi_at_train90']:.2f}, fit at {int(r['iter_train90'])}, "
                f"100% acc at {int(r['iter_train100'])}, RQI > 0.95 at {int(r['iter_rqi'])}")
    write_csv("race.csv", tab)
    r0 = [trace("released", r)["rqi"][0] for r in data["released"]]
    say(f"  released: median RQI at step 9 (initialisation) {np.median(r0):.3f}")

    # ---------------- 3. H1: movement after the fit ----------------
    say("\n3. Movement of the normalised embeddings, failures (fit between steps 1,000 and 7,000)")
    tab = []
    for t in ["released", "dec_lr_1e-3", "emb_lr_3e-3", "emb_lr_1e-2"]:
        m = movement(t, [r for r in data[t] if not r["ok"]])
        row = dict(setting=PLAIN[t], runs=len(m["net"]),
                   net_late_over_pre=med(m["net"]), path_late_over_pre=med(m["path"]),
                   grad_late_over_pre=med(m["grad"]))
        tab.append(row)
        say(f"  {PLAIN[t]}: n={row['runs']}; last 1,000 steps vs 1,000 steps before the fit: "
            f"net displacement x{row['net_late_over_pre']:.3f}, path length "
            f"x{row['path_late_over_pre']:.2f}, embedding gradient norm x{row['grad_late_over_pre']:.2f}")
    write_csv("movement.csv", tab)

    say("\n3b. When does the RQI of a failed run settle (stay within 0.05 of its final value)?")
    for t in ["released", "dec_lr_1e-3", "emb_lr_3e-3", "emb_lr_1e-2"]:
        rel, acc = [], []
        for r in data[t]:
            if r["ok"] or not r["fit"]:
                continue
            d = trace(t, r)
            rq = d["rqi"]
            bad = np.flatnonzero(np.abs(rq - rq[-1]) > 0.05)
            k = min(bad[-1] + 1 if len(bad) else 0, len(rq) - 1)
            rel.append(d["step"][k] - r["iter_train90"])
            acc.append(d["acc_train"][k])
        rel = np.array(rel)
        say(f"  {PLAIN[t]}: {len(rel)} failures; settles {np.median(rel):+.0f} steps relative to "
            f"the fit (median; IQR {np.percentile(rel, 25):+.0f} to {np.percentile(rel, 75):+.0f}); "
            f"before the fit in {np.mean(rel <= 0):.0%}; training accuracy then {np.median(acc):.2f}")

    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")

    fig_examples(data)
    fig_race(data)
    fig_movement(data)
    fig_settings(data)


def med(x):
    return float(np.median(x)) if len(x) else np.nan


def movement(tag, rows, lo=1000, hi=7000):
    """Late (last 1,000 steps) vs pre-fit (1,000 steps before the fit) ratios."""
    net, path, grad = [], [], []
    for r in rows:
        t90 = int(r["iter_train90"])
        if not (r["fit"] and lo <= t90 <= hi):
            continue
        d = trace(tag, r)
        i, j = t90 // 10, t90 // 100
        E = d["E_norm"]
        pre_net = np.linalg.norm(E[j] - E[j - 10])
        late_net = np.linalg.norm(E[-1] - E[-11])
        pre_path, late_path = d["speed_norm"][i - 100:i].sum(), d["speed_norm"][-100:].sum()
        pre_g, late_g = d["grad_emb"][i - 100:i].mean(), d["grad_emb"][-100:].mean()
        if pre_net > 0:
            net.append(late_net / pre_net)
        path.append(late_path / pre_path)
        grad.append(late_g / pre_g)
    return dict(net=net, path=path, grad=grad)


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def fig_examples(data, n=35):
    """One success and one failure of the released setting, same training set size."""
    rows = [r for r in data["released"] if r["n_train"] == n and r["fit"]]
    pick = [min((r for r in rows if r["ok"]), key=lambda r: r["seed"]),
            min((r for r in rows if not r["ok"]), key=lambda r: r["seed"])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), sharey=True)
    for ax, r, title in zip(axes, pick, ["(a) reaches RQI > 0.95", "(b) does not"]):
        d = trace("released", r)
        ax.plot(d["step"], d["acc_train"], color=INK2, lw=1.2, ls="--")
        ax.plot(d["step"], d["rqi"], color=SUCCESS if r["ok"] else FAILURE)
        t90 = r["iter_train90"]
        ax.axvline(t90, color=INK2, lw=0.8, ls=":")
        ax.text(t90 + 150, 0.04, "90% train acc.", color=INK2, fontsize=7)
        same = abs(d["acc_train"][-1] - d["rqi"][-1]) < 0.1
        ax.text(9800, d["acc_train"][-1] - (0.2 if same else 0.09), "train accuracy (dashed)",
                color=INK2, fontsize=7, ha="right")
        ax.text(9800, d["rqi"][-1] + (-0.1 if same else 0.04), "RQI (solid)", color=INK,
                fontsize=7, ha="right")
        ax.set_title(f"{title}: training set {n}/55, seed {int(r['seed'])}", loc="left")
        ax.set_xlabel("training step")
        ax.set_xlim(0, 10000)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("value")
    save(fig, "fig_examples")


def fig_race(data):
    """RQI at the fit, every lambda_3 > 0 run, by setting and outcome."""
    tags = ["dec_lr_1e-3", "released", "emb_lr_3e-3", "emb_lr_1e-2"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for k, t in enumerate(tags):
        for ok, dx, col in [(True, 0.17, SUCCESS), (False, -0.17, FAILURE)]:
            y = [r["rqi_at_train90"] for r in data[t] if r["fit"] and r["ok"] == ok]
            x = k + dx + rng.uniform(-0.1, 0.1, len(y))
            ax.scatter(x, y, s=9, color=col, alpha=0.75, linewidths=0.3, edgecolors="white",
                       label=("reaches RQI > 0.95" if ok else "does not") if k == 0 else None)
    ax.axhline(0.95, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([SETTINGS[t] for t in tags], rotation=15, ha="right")
    ax.set_ylabel("RQI at 90% training accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    save(fig, "fig_race")


def fig_movement(data, tag="released"):
    """Failures: net displacement vs path length of the normalised embeddings,
    per 500 steps, aligned to the fit."""
    offs = np.arange(-1000, 5001, 100)
    net, path = [], []
    for r in data[tag]:
        if r["ok"] or not r["fit"] or not (1000 <= r["iter_train90"] <= 4500):
            continue
        d = trace(tag, r)
        E, Es = d["E_norm"], d["E_step"]
        sp, st = d["speed_norm"], d["step"]
        t90 = r["iter_train90"]
        a, b = [], []
        for o in offs:
            t = t90 + o
            j = np.searchsorted(Es, t)
            i = np.searchsorted(st, t)
            if j + 5 < len(Es):
                a.append(np.linalg.norm(E[j + 5] - E[j]))
                b.append(sp[i:i + 50].sum())
            else:
                a.append(np.nan)
                b.append(np.nan)
        net.append(a)
        path.append(b)
    net, path = np.array(net), np.array(path)
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.plot(offs, np.nanmedian(path, 0), color=FAILURE, ls="--", lw=1.2)
    ax.plot(offs, np.nanmedian(net, 0), color=FAILURE)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("steps relative to 90% training accuracy")
    ax.set_ylabel("distance per 500 steps")
    ax.text(4900, np.nanmedian(path, 0)[-1] * 1.4, "path length", ha="right", fontsize=7)
    ax.text(2300, np.nanmedian(net, 0)[np.searchsorted(offs, 2300)] * 0.55,
            "net displacement", ha="right", fontsize=7)
    ax.set_title(f"runs that do not reach RQI > 0.95 (released, median of {len(net)})",
                 loc="left")
    save(fig, "fig_movement")


def fig_settings(data):
    """Successes per setting, with the per-seed spread."""
    tags = ["dec_lr_1e-3", "released", "emb_lr_3e-3", "emb_lr_1e-2", "dec_lr_1e-5"]
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for k, t in enumerate(tags):
        rows = data[t]
        frac = sum(r["ok"] for r in rows) / len(rows)
        ax.bar(k, frac, width=0.6, color=SUCCESS, alpha=0.35 if t != "released" else 0.6)
        n_seed = np.bincount([int(r["seed"]) for r in rows], minlength=20)
        per = per_seed_successes(rows) / n_seed
        ax.scatter(k + rng.uniform(-0.18, 0.18, 20), per, s=6, color=INK2, zorder=3,
                   linewidths=0)
        ax.text(k, 1.07, f"{sum(r['ok'] for r in rows)}/{len(rows)}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([SETTINGS[t] for t in tags], rotation=15, ha="right")
    ax.set_ylabel("fraction reaching RQI > 0.95")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", visible=False)
    save(fig, "fig_settings")


if __name__ == "__main__":
    main()
