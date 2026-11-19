"""Write every number and table that the article and the Online Appendix quote from the nested design,
the multitrait-multimethod analysis, the known-groups check and the recovery simulation.

Outputs (all generated, never edited by hand):
  results/paper/numbers.tex          one \\newcommand per quoted number
  results/paper/tab_budget.tex       article Table 1
  results/paper/tab_mtmm.tex         article Table 2
  results/paper/tab_mtmm_ci.tex      appendix MTMM table with intervals
  results/paper/tab_loio.tex         leave-one-instrument-out
  results/paper/tab_lopo.tex         leave-one-provider-out
  results/paper/tab_orientation.tex  sensitivity to the direction convention
  results/paper/tab_recovery.tex     recovery simulation
  results/paper/tab_known_groups.tex known-groups check

Run after analyze_nested_design.py, analyze_se_supplement.py, analyze_known_groups.py and
simulate_recovery.py:

    python3 make_paper_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "paper"
KEYS = [("model", "M"), ("model:instrument", "MI"), ("model:trait", "MA"), ("residual", "E")]
ROWLAB = {"model": "Model (transfers)", "model:instrument": r"Model $\times$ instrument",
          "model:trait": r"Model $\times$ axis(instrument)", "residual": "Trial noise"}
PROVIDER_NAME = {"meta-llama": "meta-llama", "mistralai": "mistralai", "x-ai": "x-ai"}
NICE = {"8values:equality": "8Values equality", "8values:progress": "8Values progress",
        "8values:liberty": "8Values liberty", "8values:peace": "8Values peace",
        "political_compass:economic": "Pol.\\ Compass economic",
        "political_compass:social": "Pol.\\ Compass social",
        "8values": "8Values", "pew_typology": "Pew Typology",
        "political_compass": "Political Compass", "sapplyvalues": "SapplyValues"}


def load(rel: str) -> dict:
    return json.loads((REPO / "results" / rel).read_text())


def pct(x: float, nd: int = 1) -> str:
    v = round(100 * x, nd)
    return f"{v + 0.0:.{nd}f}"


def ci_pct(c: dict, nd: int = 1) -> str:
    return f"[{pct(c['ci_lower'], nd)}, {pct(c['ci_upper'], nd)}]"


def tex_num(x: float, nd: int = 2, plus: bool = False) -> str:
    """A number for math mode: minus sign via '-', optional explicit plus."""
    v = round(x, nd) + 0.0
    s = f"{abs(v):.{nd}f}"
    return ("-" if v < 0 else ("+" if plus else "")) + s


def ci_num(lo: float, hi: float, nd: int = 2) -> str:
    return f"[${tex_num(lo, nd)}$, ${tex_num(hi, nd)}$]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nd = load("nested_design/nested_design.json")
    se = load("se_supplement/se_supplement.json")
    kg = load("known_groups/known_groups.json")
    sim = load("recovery_simulation/recovery_simulation.json")
    cmd: dict[str, str] = {}

    # ---- nested budget ---------------------------------------------------------------------
    for pkey, pname in (("two_instruments", "Two"), ("four_instruments", "Four")):
        ent = nd[pkey]["nested"]
        for k, short in KEYS:
            cmd[f"N{pname}{short}"] = pct(ent["point"]["share"][k])
            cmd[f"N{pname}{short}ci"] = ci_pct(ent["model_bootstrap"]["ci"][k])
            cmd[f"N{pname}{short}cip"] = ci_pct(ent["provider_bootstrap"]["ci"][k])
            cmd[f"N{pname}{short}r"] = pct(ent["point"]["share"][k], 0)
            cmd[f"N{pname}{short}cir"] = ci_pct(ent["model_bootstrap"]["ci"][k], 0)
        s = ent["point"]["share"]
        cmd[f"N{pname}Apparatus"] = pct(s["model:instrument"] + s["model:trait"], 0)
        orig = nd[pkey]["original"]
        cmd[f"O{pname}M"] = pct(orig["point"]["share"]["model"])
        cmd[f"O{pname}Mci"] = ci_pct(orig["model_bootstrap"]["ci"]["model"])
    cmd["NBoot"] = f"{nd['n_boot']:,}".replace(",", "{,}")

    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Panel & Component & Share & CI, models & CI, providers \\", r"\midrule"]
    for i, (pkey, plab) in enumerate((("two_instruments", "Two instruments"), ("four_instruments", "Four instruments"))):
        ent = nd[pkey]["nested"]
        if i:
            lines.append(r"\addlinespace")
        for j, (k, _) in enumerate(KEYS):
            lines.append(f"{plab if j == 0 else ''} & {ROWLAB[k]} & {pct(ent['point']['share'][k])} & "
                         f"{ci_pct(ent['model_bootstrap']['ci'][k])} & {ci_pct(ent['provider_bootstrap']['ci'][k])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_budget.tex").write_text("\n".join(lines) + "\n")

    # ---- leave-one-out ---------------------------------------------------------------------
    def loo_table(block: dict, label, fname: str) -> list:
        rows = [r"\begin{tabular}{lccccc}", r"\toprule",
                r"Panel & Model & M$\times$I & M$\times$A & Noise & Model, 2-term \\", r"\midrule"]
        shares = []
        for name, e in block.items():
            n = e["nested"]
            shares.append(n)
            rows.append(f"Without {label(name, e)} & {pct(n['model'])} & {pct(n['model:instrument'])} & "
                        f"{pct(n['model:trait'])} & {pct(n['residual'])} & {pct(e['original']['model'])} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        (OUT / fname).write_text("\n".join(rows) + "\n")
        return shares

    loio = loo_table(nd["leave_one_instrument_out_from_four"], lambda n, e: NICE[n], "tab_loio.tex")
    lopo = loo_table(nd["leave_one_provider_out_four_instruments"],
                     lambda n, e: f"{n} ({e['n_models']})", "tab_lopo.tex")
    for tag, shares in (("Loio", loio), ("Lopo", lopo)):
        for k, short in KEYS[:3]:
            vals = [s[k] for s in shares]
            cmd[f"{tag}{short}min"], cmd[f"{tag}{short}max"] = pct(min(vals)), pct(max(vals))

    # ---- direction convention --------------------------------------------------------------
    vlab = {"aligned": "Aligned (reported)", "peace_reversed": "8Values peace reversed",
            "sapply_auth_reversed": "SapplyValues authority reversed",
            "peace_and_sapply_auth_reversed": "Both reversed",
            "unaligned_as_scored": "No alignment (scores as each instrument reports them)"}
    sens = nd["orientation_sensitivity_nested"]
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"Panel & Direction convention & Model & M$\times$I & M$\times$A & Noise \\", r"\midrule"]
    for i, (pkey, plab) in enumerate((("two_instruments", "Two instruments"), ("four_instruments", "Four instruments"))):
        if i:
            rows.append(r"\addlinespace")
        first = True
        for v, lab in vlab.items():
            if pkey == "two_instruments" and "sapply" in v:
                continue  # SapplyValues is not in the two-instrument panel
            s = sens[pkey][v]
            rows.append(f"{plab if first else ''} & {lab} & {pct(s['model'])} & {pct(s['model:instrument'])} & "
                        f"{pct(s['model:trait'])} & {pct(s['residual'])} \\\\")
            first = False
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_orientation.tex").write_text("\n".join(rows) + "\n")
    for pkey, pname in (("two_instruments", "Two"), ("four_instruments", "Four")):
        alt = [sens[pkey][v]["model"] for v in vlab if v not in ("aligned", "unaligned_as_scored")
               and not (pkey == "two_instruments" and "sapply" in v)]
        cmd[f"Orient{pname}Min"], cmd[f"Orient{pname}Max"] = pct(min(alt)), pct(max(alt))
        cmd[f"Unaligned{pname}M"] = pct(sens[pkey]["unaligned_as_scored"]["model"])

    # ---- MTMM ------------------------------------------------------------------------------
    pairs = se["mtmm_pairs"]["pairs"]
    mm = se["mtmm_means"]
    conv = [p for p in pairs if p["type"] == "convergent"]
    econ = next(p for p in conv if "economic" in p["b"] or "economic" in p["a"])
    soc = next(p for p in conv if "social" in p["b"] or "social" in p["a"])
    cmd["ConvEcon"], cmd["ConvEconSE"] = tex_num(econ["r"], 3, True), f"{econ['se_r']:.2f}"
    cmd["ConvSoc"], cmd["ConvSocSE"] = tex_num(soc["r"], 3, True), f"{soc['se_r']:.2f}"
    for key, name in (("mean_convergent", "Conv"), ("mean_discriminant", "Disc"),
                      ("convergent_minus_discriminant", "Diff")):
        m = mm[key]
        cmd[f"{name}Mean"] = tex_num(m["estimate"], 2, name != "Diff")
        cmd[f"{name}Abs"] = f"{abs(m['estimate']):.2f}"
        cmd[f"{name}SE"] = f"{m['se']:.2f}"
        cmd[f"{name}CI"] = ci_num(m["ci_lower"], m["ci_upper"])
    cmd["DiffShareBelow"] = pct(mm["share_of_resamples_convergent_below_discriminant"])
    ev = se["eightvalues_internal"]["mean_r"]
    cmd["EightInternal"], cmd["EightInternalSE"] = f"{ev['estimate']:.2f}", f"{ev['se']:.2f}"
    cmd["EightInternalCI"] = ci_num(ev["ci_lower"], ev["ci_upper"])

    def pair_label(p):
        return f"{NICE[p['a']]} vs.\\ {NICE[p['b']]}"

    body, body_ci = [], []
    for p in pairs:
        typ = p["type"].capitalize()
        body.append(f"{typ} & {pair_label(p)} & ${tex_num(p['r'], 3, True)}$ & {p['se_r']:.3f} \\\\")
        body_ci.append(f"{typ} & {pair_label(p)} & ${tex_num(p['r'], 3, True)}$ & {p['se_r']:.3f} & "
                       f"{ci_num(*p['ci_fisher'])} \\\\")
    means, means_ci = [], []
    for key, lab in (("mean_convergent", "Mean convergent"), ("mean_discriminant", "Mean discriminant"),
                     ("convergent_minus_discriminant", "Difference")):
        m = mm[key]
        means.append(f"{lab} & & ${tex_num(m['estimate'], 3, True)}$ & {m['se']:.3f} \\\\")
        means_ci.append(f"\\textbf{{{lab}}} & & ${tex_num(m['estimate'], 3, True)}$ & {m['se']:.3f} & "
                        f"{ci_num(m['ci_lower'], m['ci_upper'])} \\\\")
    (OUT / "tab_mtmm.tex").write_text("\n".join(
        [r"\begin{tabular}{llrr}", r"\toprule", r"Type & Pair & $r$ & $SE$ \\", r"\midrule"] + body
        + [r"\midrule"] + means + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    (OUT / "tab_mtmm_ci.tex").write_text("\n".join(
        [r"\begin{tabular}{lp{6.4cm}rrr}", r"\toprule", r"Type & Pair & $r$ & $SE$ & 95\% CI \\", r"\midrule"]
        + body_ci + [r"\midrule"] + means_ci + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    # ---- Spearman checks on the added instruments ------------------------------------------
    names = {("sapplyvalues:right", "political_compass:economic"): "RhoSapPC",
             ("sapplyvalues:right", "8values:equality"): "RhoSapEq",
             ("sapplyvalues:auth", "political_compass:social"): "RhoAuthPC",
             ("sapplyvalues:auth", "8values:liberty"): "RhoAuthLib",
             ("sapplyvalues:prog", "8values:progress"): "RhoProgProg",
             ("pew_typology:position", "political_compass:economic"): "RhoPewPC",
             ("pew_typology:position", "8values:equality"): "RhoPewEq"}
    for s in se["spearman"]:
        n = names[(s["a"], s["b"])]
        cmd[n], cmd[n + "SE"] = tex_num(s["estimate"], 2, True), f"{s['se']:.2f}"
        cmd[n + "CI"] = ci_num(s["ci_lower"], s["ci_upper"])

    # ---- known groups ----------------------------------------------------------------------
    cmd["KGModels"], cmd["KGCells"] = str(kg["n_models"]), str(kg["n_cells"])
    cmd["KGPositive"] = str(round(kg["share_of_cells_positive"] * kg["n_cells"]))
    rng_shares = [a["mean_gap_share_of_range"] for a in kg["per_axis"].values()]
    cmd["KGRangeLo"], cmd["KGRangeHi"] = pct(min(rng_shares), 0), pct(max(rng_shares), 0)
    lo, hi = kg["gap_in_between_model_sd_range"]
    cmd["KGSdLo"], cmd["KGSdHi"] = f"{lo:.0f}", f"{hi:.0f}"
    for key, name in (("asker_shift_correlations", "KGShift"), ("no_asker_position_correlations", "KGBase")):
        b = kg[key]
        cmd[name] = tex_num(b["mean_convergent"]["estimate"], 2)
        cmd[name + "CI"] = ci_num(b["mean_convergent"]["ci_lower"], b["mean_convergent"]["ci_upper"])
        cmd[name + "All"] = tex_num(b["mean_all_cross_instrument"]["estimate"], 2)
        cmd[name + "AllLo"], cmd[name + "AllHi"] = (tex_num(x, 2) for x in b["cross_instrument_range"])
        for (pk, r), nm in zip(b["convergent_pairs"].items(), ("Econ", "Soc")):
            cmd[name + nm] = tex_num(r, 2)
    d = kg["convergent_shift_minus_position"]
    cmd["KGDiff"], cmd["KGDiffCI"] = tex_num(d["estimate"], 2), ci_num(d["ci_lower"], d["ci_upper"])
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"Axis & Gap (95\% CI) & \% of range & Between-model SDs & Models shifting left \\", r"\midrule"]
    for t, a in kg["per_axis"].items():
        g = a["mean_gap"]
        rows.append(f"{NICE[t]} & {g['estimate']:.1f} [{g['ci_lower']:.1f}, {g['ci_upper']:.1f}] & "
                    f"{pct(a['mean_gap_share_of_range'], 0)} & {a['mean_gap_in_between_model_sd']:.1f} & "
                    f"{round(a['share_of_models_positive'] * kg['n_models'])} of {kg['n_models']} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_known_groups.tex").write_text("\n".join(rows) + "\n")

    # ---- recovery simulation ---------------------------------------------------------------
    cmd["SimReps"] = str(sim["reps"])
    rows = [r"\begin{tabular}{llccccc}", r"\toprule",
            r"Panel & True share & Mean & Median & 5th--95th pct. & Est.\ $<1\%$ & Est.\ $\le$ observed \\",
            r"\midrule"]
    for i, (pkey, plab, pname) in enumerate((("two_instruments", "Two instruments", "Two"),
                                             ("four_instruments", "Four instruments", "Four"))):
        p = sim["panels"][pkey]
        if i:
            rows.append(r"\addlinespace")
        for j, (truth, r) in enumerate(p["by_true_share"].items()):
            rows.append(f"{plab if j == 0 else ''} & {pct(float(truth), 0)} & {pct(r['mean'])} & {pct(r['median'])} & "
                        f"[{pct(r['q05'])}, {pct(r['q95'])}] & {pct(r['share_below_1pct'], 0)} & "
                        f"{pct(r['share_at_or_below_observed'], 0)} \\\\")
            tag = {"0.00": "Zero", "0.02": "Two", "0.05": "Five", "0.10": "Ten", "0.20": "Twenty", "0.30": "Thirty"}[truth]
            cmd[f"Sim{pname}{tag}Below"] = pct(r["share_at_or_below_observed"], 0)
            cmd[f"Sim{pname}{tag}Above"] = pct(r["share_at_or_above_observed"], 0)
            cmd[f"Sim{pname}{tag}BelowN"] = str(round(r["share_at_or_below_observed"] * sim["reps"]))
            cmd[f"Sim{pname}{tag}AboveN"] = str(round(r["share_at_or_above_observed"] * sim["reps"]))
            cmd[f"Sim{pname}{tag}Mean"] = pct(r["mean"])
            cmd[f"Sim{pname}{tag}Near"] = pct(r["share_below_1pct"], 0)
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_recovery.tex").write_text("\n".join(rows) + "\n")

    # ---- reliability-corrected correlations (analyze_reliability_corrected.py) -------------------
    rc_path = REPO / "results" / "reliability_corrected" / "reliability_corrected.json"
    if rc_path.exists():
        rc = json.loads(rc_path.read_text())
        rels = list(rc["reliability_of_model_means"].values())
        cmd["RelMin"] = f"{min(rels):.2f}"
        cmd["RelMax"] = f"{max(rels):.2f}"
        cmd["ConvMeanCorr"] = tex_num(rc["point"]["corrected"]["conv_mean"], 2, plus=True)
        cmd["DiffCorr"] = tex_num(rc["point"]["corrected"]["diff"], 2)
        cmd["EconCorr"] = tex_num(rc["point"]["corrected"]["conv"][0], 3, plus=True)
        lo, hi = rc["ci"]["corrected"]["conv_pair_economic"]
        cmd["EconCorrCI"] = ci_num(lo, hi)

    head = "% Generated by scoring/make_paper_numbers.py from the result files. Do not edit by hand.\n"
    (OUT / "numbers.tex").write_text(
        head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(cmd.items())))
    print(f"wrote {len(cmd)} numbers and 8 tables to {OUT}")


if __name__ == "__main__":
    main()
