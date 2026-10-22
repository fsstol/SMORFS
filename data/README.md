# Data

SEER's data-use agreement does not allow SEER data, or files derived from it, in a public repository. This folder
therefore holds only the 2021 period life table of the US Social Security Administration (Table 4.C6). The three
SEER-derived files the bladder-cancer model reads must be made by each user from SEER under their own data-use
agreement, as described below. Without them, the code runs on **synthetic** data (`tools/synthetic_data.py`, see the
main README), which has the same files and formats.

**Never commit SEER files.** `.gitignore` ignores `data/single-multiple/` and the case-level exports named below.

## Included: `US_life_table_2021.csv`

The 2021 period life table of the US Social Security Administration (Table 4.C6, both sexes,
https://www.ssa.gov/oact/STATS/table4c6.html), public data. Two columns:

| Column | Meaning |
|---|---|
| `Age` | exact age in years, 0 to 119 |
| `Total` | survivors at that age per 100,000 born |

`parameters.get_survival_function()` turns it into the all-cause survival S_US(t) on the model's time grid.

## Expected, not included: `single-multiple/`

| File | What it holds | Target family it gives |
|---|---|---|
| `rates_urinary_bladder.csv` | counts and populations of diagnosed cases | clinical incidence |
| `mortality_urinary_bladder.csv` | counts and populations of deaths from this cancer after diagnosis | clinical mortality |
| `doa_urinary bladder.pkl` (the name contains a space) | counts and populations of deaths within 2 months of diagnosis, from this cancer (outcome 1) or other causes (outcome 2) | preclinical mortality and preclinical other-cause mortality |

For each family, stage group and single year of age 35 to 80 (552 targets in all), `main.py` takes the target as
count / person-years, with standard error sqrt(count) / person-years. Its functions are `get_target_incidence`,
`get_target_mortality` and `get_target_doa`, and `verification/h2.py`'s `exact_targets` calls them.
- **Counts** are summed over the SEER stages of a group: Early = "In situ" + "Localized", Medium = "Regional",
  Late = "Distant".
- **The person-years** are the same for all four families: SEER 17's for the years of diagnosis 2004 to 2021, by
  single year of age, summed from the rates file's single-year rows (`main.seer17_person_years`). In SEER*Stat rate
  sessions every stage row of a year and age carries the same population, so one stage's rows are used. The
  mortality file's and the pickle's own `Population` columns are not used (`main.corrected_denominators = True`, the
  default since v1.0.1, `CORRECTIONS.md` entry 16; `False` gives the targets of v1.0.0 and of the first submission).
- **Rows that are dropped:** stage "Blank(s)" (or missing) and "Unknown/unstaged", the year-of-diagnosis aggregate
  "2000-2021", and the ages outside 35 to 80.

### `rates_urinary_bladder.csv`: a SEER*Stat rate session, exported as CSV

Columns, in this order:

| Column | Values |
|---|---|
| `Combined Summary Stage (2004+)` | "In situ", "Localized", "Regional", "Distant", "Unknown/unstaged", "Blank(s)" |
| `Year of diagnosis` | single years 2000 to 2021, and the aggregate "2000-2021" (which `main.py` drops) |
| `Age recode with single ages and 85+` | "00 years" to "84 years", "85+ years" |
| `Crude Rate`, `Standard Error`, `Lower Confidence Interval`, `Upper Confidence Interval` | per 100,000, as SEER*Stat prints them ("~" for an undefined SE); not used by the model |
| `Count`, `Population` | integers |

- SEER*Stat database: Incidence - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)
- Years of diagnosis: 2004 to 2021. The file lists every year of the database, 2000 to 2021, but the rows of 2000 to
  2003 have count and population 0, so the incidence targets' cases and person-years are those of 2004 to 2021
  (`main.py` keeps all single years and drops the aggregate)
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior
- Session settings: rate session, statistic = crude rates, standard errors and 95% CIs

### `mortality_urinary_bladder.csv`: a SEER*Stat session of deaths after diagnosis, exported as CSV

Columns:

| Column | Values |
|---|---|
| `Combined Summary Stage (2004+)` | as above |
| `SEER cause-specific death classification` | `main.py` keeps "Dead (attributable to this cancer dx)" only |
| `Age at death with single ages and 85+` | "00 years" to "84 years", "85+ years" |
| `Crude Rate`, `Standard Error`, `Lower Confidence Interval`, `Upper Confidence Interval` | per 100,000; not used by the model |
| `Count`, `Population` | integers |

- SEER*Stat database: Incidence-Based Mortality - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)
- Years: the file has no year column. Its deaths and person-years are those of the years of death 2000 to 2021, the
  database's whole span, for cases diagnosed from 2000. "Combined Summary Stage (2004+)" is defined only for cases
  diagnosed from 2004, so the deaths of cases diagnosed in 2000 to 2003 have stage "Blank(s)"; `main.py` drops them,
  and they enter no target. The deaths that remain are of cases diagnosed from 2004, and `main.py` divides them by
  the rates file's person-years of 2004 to 2021, not by this file's population of 2000 to 2021
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior
- Session settings: rate session, statistic = crude rates, standard errors and 95% CIs

### `doa_urinary bladder.pkl`: built by `main.py` from a case listing

A pandas DataFrame, pickled, with index (`outcome`, `Combined Summary Stage (2004+)`,
`Age recode with single ages and 85+`) and integer columns `Population` and `Count`:
- `outcome` is 1 for death from this cancer and 2 for death from other causes;
- the stages are "Distant", "In situ", "Localized" and "Regional";
- the ages are as above.

It is built by `main.py`'s `main_pickle()` and `main_doa()`, which are not called by default:
1. **A case listing.** Export from SEER*Stat as `single-multiple/single.csv`; `main_pickle()` converts it to
   `single.pkl`. It is **case-level data: keep it out of every repository.** Columns used:
   - `Site recode ICD-O-3/WHO 2008`;
   - `Vital status recode (study cutoff used)`;
   - `SEER cause-specific death classification` and `SEER other cause of death classification`;
   - `Survival months`;
   - `Combined Summary Stage (2004+)`;
   - `Year of death recode`;
   - `Age recode with single ages and 85+`.
2. **The populations.** `single-multiple/populations.csv`, with columns `Year`,
   `Age recode with single ages and 90+` and `Count`. `main_doa()` keeps the years 2004 to 2021, and groups ages 85 to
   89 and 90+ as "85+ years". These populations go into the pickle's `Population` column, which `main.py`'s targets
   no longer use: since v1.0.1 the deaths at diagnosis, which are SEER 17's cases, are divided by SEER 17's person-years
   of 2004 to 2021 from the rates file, like every other family (`CORRECTIONS.md` entry 16).
   - SEER*Stat database, as reported by the author who made the file: Populations - Total U.S. (1990-2023)
     <Katrina/Rita Adjustment>. The populations in the pickle do not match the total US, however: by single year of
     age, they match the SEER 22 area (SEER 17 plus Idaho, Illinois, Massachusetts, New York and Texas) for 2004 to
     2021, against the Census Bureau's population estimates. Either way they are not SEER 17's, which is why they
     are no longer used.
3. **The deaths.** `main_doa()` counts, by outcome, stage and age, the deaths with survival of at most 2 months
   (`parameters.months_cutoff`). It excludes cases with unknown survival or "Unknown/unstaged" stage.

- SEER*Stat database: Incidence - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)
- Years of diagnosis: 2004 to 2021; `main_doa()` keeps the populations of the same years
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior; known age
- Session settings: case listing session

## Rebuilding the targets and checking them

With the three files in place, `python tests/table2_mle.py` rebuilds the targets with `main.py`'s functions, fits the
model, and compares Table 2's MLE column with `output/final/urinary bladder CIs.csv` (exit status 1 if any cell
differs). `main.py`'s run also writes the targets it uses to `output/urinary bladder_rates.csv`. That file is
SEER-derived: do not commit it.
