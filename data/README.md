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
count / population, with standard error sqrt(count) / population. Its functions are `get_target_incidence`,
`get_target_mortality` and `get_target_doa`, and `verification/h2.py`'s `exact_targets` calls them.
- **Counts** are summed over the SEER stages of a group: Early = "In situ" + "Localized", Medium = "Regional",
  Late = "Distant".
- **The population** is the group's first stage's. In SEER*Stat rate sessions every stage row carries the same
  population.
- **Rows that are dropped:** stage "Blank(s)" (or missing) and "Unknown/unstaged", the year-of-diagnosis aggregate
  "2000-2021", and the ages outside 35 to 80.

### `rates_urinary_bladder.csv`: a SEER*Stat rate session, exported as CSV

Columns, in this order:

| Column | Values |
|---|---|
| `Combined Summary Stage (2004+)` | "In situ", "Localized", "Regional", "Distant", "Unknown/unstaged", "Blank(s)" |
| `Year of diagnosis` | single years, and the aggregate "2000-2021" (which `main.py` drops) |
| `Age recode with single ages and 85+` | "00 years" to "84 years", "85+ years" |
| `Crude Rate`, `Standard Error`, `Lower Confidence Interval`, `Upper Confidence Interval` | per 100,000, as SEER*Stat prints them ("~" for an undefined SE); not used by the model |
| `Count`, `Population` | integers |

- SEER*Stat database: Incidence - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)"
- Years of diagnosis: 2004 to 2021, as single years, plus the aggregate
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior
- Session settings: rate session, statistic = crude rates, standard errors and 95% CIs; 2000 US Std Population (single ages to 84 - Census P25-1130)

### `mortality_urinary_bladder.csv`: a SEER*Stat session of deaths after diagnosis, exported as CSV

Columns:

| Column | Values |
|---|---|
| `Combined Summary Stage (2004+)` | as above |
| `SEER cause-specific death classification` | `main.py` keeps "Dead (attributable to this cancer dx)" only |
| `Age at death with single ages and 85+` | "00 years" to "84 years", "85+ years" |
| `Crude Rate`, `Standard Error`, `Lower Confidence Interval`, `Upper Confidence Interval` | per 100,000; not used by the model |
| `Count`, `Population` | integers |

- SEER*Stat database: Incidence-Based Mortality - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)"
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior
- Session settings: rate session, statistic = crude rates, standard errors and 95% CIs; 2000 US Std Population (single ages to 84 - Census P25-1130)

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
   89 and 90+ as "85+ years".
3. **The deaths.** `main_doa()` counts, by outcome, stage and age, the deaths with survival of at most 2 months
   (`parameters.months_cutoff`). It excludes cases with unknown survival or "Unknown/unstaged" stage.

- SEER*Stat database: Incidence - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)"
- Years of diagnosis: 2004 to 2021, as single years
- Case selection: Site recode ICD-O-3/WHO 2008 = Urinary Bladder; malignant behavior; known age
- Session settings: case listing session

## Rebuilding the targets and checking them

With the three files in place, `python tests/table2_mle.py` rebuilds the targets with `main.py`'s functions, fits the
model, and compares Table 2's MLE column with `output/final/urinary bladder CIs.csv` (exit status 1 if any cell
differs). `main.py`'s run also writes the targets it uses to `output/urinary bladder_rates.csv`. That file is
SEER-derived: do not commit it.
