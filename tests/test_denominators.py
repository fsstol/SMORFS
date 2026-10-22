"""denominators: every target's person-years are the rates file's, years 2004 to 2021 (CORRECTIONS.md, entry 16).

Needs the three data files in data/single-multiple/; in the public suite they are the SYNTHETIC ones
(tools/synthetic_data.py, which gives the mortality file and the pickle their own, different Population columns, as the
SEER files have). Without the files it is reported as info. The targets are built with main.py's functions
(verification/h2.py's exact_targets), with main.corrected_denominators True (the default) and False.
Gates:
  - the clinical incidence targets are the same with both settings;
  - with the default, every clinical mortality and deaths-at-diagnosis target with events is count / person-years and
    its SE sqrt(count) / person-years, where the counts are summed over the group's stages and the person-years are
    the rates file's for 2004 to 2021, computed here from the files directly (one stage's rows, summed over the
    years); a target with no events keeps weight 0;
  - with False, those families divide by the files' own populations instead (the original analysis);
  - main.seer17_person_years raises if a year of 2004 to 2021 is missing from the rates file.
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, build, load_h2, row

DATA = os.path.join(ROOT, "data", "single-multiple")
STAGE = "Combined Summary Stage (2004+)"
AGE = "Age recode with single ages and 85+"
AGE_DEATH = "Age at death with single ages and 85+"
FAMILIES = {"clinical mortality": None, "preclinical mortality": 1, "preclinical other mortality": 2}


def run(ctx):
    files = ["rates_urinary_bladder.csv", "mortality_urinary_bladder.csv", "doa_urinary bladder.pkl"]
    if not all(os.path.exists(os.path.join(DATA, f)) for f in files):
        return [row("denominators", "targets' person-years", "no data files", "info", None)]
    rates = pd.read_csv(os.path.join(DATA, files[0]))
    mort = pd.read_csv(os.path.join(DATA, files[1]))
    mort = mort[mort["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)"]
    doa = pd.read_pickle(os.path.join(DATA, files[2])).reset_index()
    years = pd.to_numeric(rates["Year of diagnosis"], errors="coerce")
    one = rates[years.between(2004, 2021) & (rates[STAGE] == "Localized")]
    py = one.groupby(AGE)["Population"].sum()
    own = {"clinical mortality": mort.groupby(AGE_DEATH)["Population"].first(),
           "preclinical mortality": doa.groupby(AGE)["Population"].first(),
           "preclinical other mortality": doa.groupby(AGE)["Population"].first()}

    h2 = load_h2()
    new, old = build(h2, True), build(h2, False)
    groups = new.stage_groups
    rows = [row("denominators", "clinical incidence targets, corrected vs original setting",
                "identical" if new.targets["clinical incidence"] == old.targets["clinical incidence"] else "differ",
                "identical", new.targets["clinical incidence"] == old.targets["clinical incidence"])]
    worst_new, worst_old, n, zero_ok = 0.0, 0.0, 0, True
    for family, outcome in FAMILIES.items():
        for group, stages in groups.items():
            if outcome is None:
                c = mort[mort[STAGE].isin(stages)].groupby(AGE_DEATH)["Count"].sum()
            else:
                c = doa[(doa["outcome"] == outcome) & doa[STAGE].isin(stages)].groupby(AGE)["Count"].sum()
            for a in h2.P.ages_eval:
                count = int(c.get(a, 0))
                mu, se = new.targets[family][group][a]
                mu0, se0 = old.targets[family][group][a]
                if count == 0:
                    zero_ok &= mu == 0.0 and se >= 1e19
                    continue
                n += 1
                worst_new = max(worst_new, abs(mu / (count / py[a]) - 1), abs(se / (np.sqrt(count) / py[a]) - 1))
                worst_old = max(worst_old, abs(mu0 / (count / own[family][a]) - 1),
                                abs(se0 / (np.sqrt(count) / own[family][a]) - 1))
    rows.append(row("denominators", "default: count / rates file's 2004-2021 person-years, max rel diff", worst_new,
                    "<= 1e-12", worst_new <= 1e-12, f"{n} mortality and deaths-at-diagnosis targets with events"))
    rows.append(row("denominators", "default: targets with no events keep weight 0", zero_ok, "True", zero_ok))
    rows.append(row("denominators", "corrected_denominators = False: the files' own populations, max rel diff",
                    worst_old, "<= 1e-12", worst_old <= 1e-12))
    ages = sorted(h2.P.ages_eval)
    ratio = float(np.max(np.abs(own["preclinical mortality"][ages] / py[ages] - 1)))
    rows.append(row("denominators", "the two settings' deaths-at-diagnosis denominators differ (max rel)", ratio,
                    "info", None, "0 would make the check above vacuous"))
    try:
        h2.M.seer17_person_years(rates[pd.to_numeric(rates["Year of diagnosis"], errors="coerce") != 2010])
        raised = False
    except ValueError:
        raised = True
    rows.append(row("denominators", "a missing year of diagnosis raises", "raises" if raised else "no error",
                    "raises ValueError", raised))
    return rows
