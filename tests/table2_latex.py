"""Table 2 as LaTeX, from the summary table (entry 84).

    python tests/table2_latex.py [--csv "output/final/urinary bladder CIs.csv"] [--out output/final/table2.tex]

The cells are the summary table's strings as main_rates writes them (four decimal places, entry 76): the MLE with its
95% CI from the observed information, and the LIMIS median with its 95% CrI. The layout, descriptions and caption are
the paper's Table 2 as one of the authors supplied it (the caption's "palces" corrected).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

ROWS = [  # (summary table index, LaTeX parameter, description)
    ("Incidence_1", r"Incidence$_1$", r"\small Weibull incidence: shape parameter"),
    ("Incidence_2", r"Incidence$_2$", r"\small Weibull incidence: scale parameter (years)"),
    ("U1->U2", r"$U_1 \to U_2$", r"\small \makecell[l]{Early untreated to medium untreated: \\ constant hazard rate}"),
    ("U1->T1", r"$U_1 \to T_1$", r"\small \makecell[l]{Early untreated to treated: \\ constant hazard rate}"),
    ("U1->D1", r"$U_1 \to D_c$", r"\small \makecell[l]{Early untreated cancer death: \\ constant hazard rate}"),
    ("U2->U3", r"$U_2 \to U_3$", r"\small \makecell[l]{Medium untreated to late untreated: \\ constant hazard rate}"),
    ("U2->T2", r"$U_2 \to T_2$", r"\small \makecell[l]{Medium untreated to treated: \\ constant hazard rate}"),
    ("U2->D2", r"$U_2 \to D_c$", r"\small \makecell[l]{Medium untreated cancer death: \\ constant hazard rate}"),
    ("U3->T3", r"$U_3 \to T_3$", r"\small \makecell[l]{Late untreated to treated: \\ constant hazard rate}"),
    ("U3->D3", r"$U_3 \to D_c$", r"\small \makecell[l]{Late untreated cancer death: \\ constant hazard rate}"),
    ("T1->D1", r"$T_1 \to D_c$", r"\small \makecell[l]{Early treated cancer death: \\ constant hazard rate}"),
    ("T2->D2", r"$T_2 \to D_c$", r"\small \makecell[l]{Medium treated cancer death: \\ constant hazard rate}"),
    ("T3->D3", r"$T_3 \to D_c$", r"\small \makecell[l]{Late treated cancer death: \\ constant hazard rate}"),
]
HEAD = r"""\begin{table}[!ht]
    \centering
    \resizebox{\textwidth}{!}{
    \begin{tabular}{|l|l|l|l|}
    \hline
        \textbf{Parameter} & \textbf{Description} & \textbf{MLE (CI)} & \textbf{LIMIS (CrI)} \\ \hline
"""
TAIL = r"""    \end{tabular}
    }
    \caption{Model parameters: maximum likelihood estimates with 95\% confidence intervals (CI) from the observed information, $\exp(\log\hat\beta \pm 1.96\,\mathrm{SE})$, and posterior medians with 95\% credible intervals (CrI), the 2.5\% and 97.5\% weighted quantiles of the 20 pooled LIMIS runs. Rates are per year; values are rounded to four decimal places. MLE: maximum likelihood estimation; LIMIS: Langevin incremental mixture importance sampling.}
    \label{tab:parameters}
\end{table}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(ROOT, "output", "final", "urinary bladder CIs.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "final", "table2.tex"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv, index_col=0)
    if sorted(df.index) != sorted(k for k, _, _ in ROWS):
        sys.exit(f"unexpected rows in {args.csv}: {list(df.index)}")
    body = "".join(f"        {p} & {d} & {df.loc[k, 'mle_str']} & {df.loc[k, 'imis_str']} \\\\ \\hline\n"
                   for k, p, d in ROWS)
    with open(args.out, "w") as fh:
        fh.write(HEAD + body + TAIL)
    print(body, end="")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
