import pandas as pd
import numpy as np
import string
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from typing import Optional, Dict, List, Tuple

# Maryland-only post-stratification weights (per model sample)
# Calibration cells: SurveyYear × geo_bin × race_bin × age_bin
# geo_bin: Central vs Outlying (Outside_CBSA_or_Unclassified -> Outlying)
# race_bin: non-hispanic white / non-hispanic black / hispanic / other
# age_bin: 18-34 / 35-54 / 55+

"""
Census ACS 5-Year API for Maryland County-Level Weighting Targets
API docs: https://api.census.gov/data.html
"""

# Helper Functions

def _norm_text(s: str) -> str:
    return str(s).strip() if not pd.isna(s) else np.nan

def map_geo_bin(x: str) -> str:
    if pd.isna(x):
        return np.nan
    x = _norm_text(x)
    if x == "Central":
        return "Central"
    if x in ("Outlying", "Outside_CBSA_or_Unclassified"):
        return "Outlying"
    return np.nan

def clean_md_county_name(x: str) -> str:
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace(" County", "").replace(" county", "")
    if x.lower() in ["baltimore city", "baltimore city."]:
        return "Baltimore City"
    if x.lower() in ["saint mary's", "saint marys", "saint mary's city", "saint mary"]:
        return "St. Mary's"
    if x.lower() in ["st. mary's", "st mary's", "st marys"]:
        return "St. Mary's"
    if x.lower() in ["prince george's", "prince georges", "prince george’s"]:
        return "Prince George's"
    x = x.replace("CITY", "City").replace("city", "City")
    if x.lower() in ["baltimore city", "baltimorecity"]:
        return "Baltimore City"
    return x

def map_age_bin(age_clean: pd.Series) -> pd.Series:
    m = {
        "18-24": "18-34",
        "25-34": "18-34",
        "35-44": "35-54",
        "45-54": "35-54",
        "55-64": "55+",
        "65_plus": "55+",
    }
    return age_clean.map(m)

def map_race_bin(race_clean: pd.Series) -> pd.Series:
    def _f(x):
        if pd.isna(x):
            return np.nan
        x = _norm_text(x)
        if x == "white":
            return "non-hispanic white"
        if x == "black":
            return "non-hispanic black"
        if x == "hispanic_latinx":
            return "hispanic"
        if x in ("asian", "multi_racial_other"):
            return "other"
        if x.lower() in ("unknown", "prefer not to identify", "pnta"):
            return np.nan
        return "other"
    return race_clean.apply(_f)
    
# 2) Build ACS population targets aggregated to geo × race × age
    
def build_pop_targets_md(
    county_employed_tbl: pd.DataFrame,
    county_to_geo: pd.Series,
    target_counties: list[str],
    year_levels: list[str]
) -> pd.DataFrame:
    """
    county_employed_tbl: table indexed by county names, with columns as a MultiIndex of race_bin and age_bin.
    county_to_geo: whether a county is classified as 'Central' or 'Outlying.'
    target_counties: all counties in the weighting.
    year_levels: years in the model sample.
    """
    pop = county_employed_tbl.copy()
    pop.index = pop.index.map(clean_md_county_name)
    target_counties = [clean_md_county_name(c) for c in target_counties]
    target_counties = sorted({c for c in target_counties if c in set(pop.index)})
    pop = pop.loc[pop.index.isin(target_counties)].copy()
    county_to_geo = county_to_geo.copy()
    county_to_geo.index = county_to_geo.index.map(clean_md_county_name)
    pop["geo_bin"] = county_to_geo.reindex(pop.index)
    pop["geo_bin"] = pop["geo_bin"].fillna("Outlying")
    race_age_cols = [c for c in pop.columns if isinstance(c, tuple)]
    pop_geo = pop.groupby("geo_bin")[race_age_cols].sum()
    pop_long = (
        pop_geo.stack(level=[0, 1])
              .rename("n_pop")
              .reset_index()
    )
    pop_long.columns = ["geo_bin", "race_bin", "age_bin", "n_pop"]
    pop_long = pd.concat(
        [pop_long.assign(SurveyYear=str(y)) for y in year_levels],
        ignore_index=True
    )
    return pop_long
    
# 3) Compute post-strat weights per model sample (Maryland-only)

def compute_md_poststrat_weights(
    df_model: pd.DataFrame,
    county_employed_tbl: pd.DataFrame,
    *,
    year_col="SurveyYear",
    state_col="home_state",
    county_col="home_county",
    geo_col="home_central_outlying_county",
    race_col="race_clean",
    age_col="age_clean",
 #   md_state_value="Maryland",
    trim_q=(0.01, 0.99),
    max_cap=10.0,
    target_universe="model_counties"  
) -> pd.DataFrame:
    """
      - w_raw: post-strat weight (year-normalized)
      - w_trim: trimmed/capped weight used for reporting
      - w_in_calibration: 1 if row had complete bins and is MD resident, else 0
      - calibration bin columns: geo_bin, race_bin, age_bin
    """
    df = df_model.copy()
    df["county_clean"] = df[county_col].map(clean_md_county_name)
    df["geo_bin"] = df[geo_col].map(map_geo_bin)
    df["race_bin"] = map_race_bin(df[race_col])
    df["age_bin"] = map_age_bin(df[age_col])
    df[year_col] = df[year_col].astype(str)
    
    if target_universe == "model_counties":
        target_counties = sorted(df["county_clean"].dropna().unique().tolist())
    elif target_universe == "all_md_counties_in_acs":
        target_counties = sorted(pd.Index(county_employed_tbl.index).map(clean_md_county_name).dropna().unique().tolist())
    else:
        raise ValueError("target_universe must be 'model_counties' or 'all_md_counties_in_acs'")

    county_to_geo = (
        df.dropna(subset=["county_clean"])
          .assign(geo_bin=df["geo_bin"].fillna("Outlying"))
          .groupby("county_clean")["geo_bin"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Outlying")
    )
    year_levels = sorted(df[year_col].dropna().unique().tolist())

    pop_targets = build_pop_targets_md(
        county_employed_tbl=county_employed_tbl,
        county_to_geo=county_to_geo,
        target_counties=target_counties,
        year_levels=year_levels
    )

    df["in_calibration"] = (
        df[year_col].notna() &
        df["geo_bin"].notna() &
        df["race_bin"].notna() &
        df["age_bin"].notna()
    )

    sample_cells = (
        df.loc[df["in_calibration"]]
          .groupby([year_col, "geo_bin", "race_bin", "age_bin"])
          .size()
          .rename("n_sample")
          .reset_index()
    )

    cells = sample_cells.merge(
        pop_targets,
        on=[year_col, "geo_bin", "race_bin", "age_bin"],
        how="left"
    )

    if cells["n_pop"].isna().any():
        miss = cells.loc[cells["n_pop"].isna(), [year_col, "geo_bin", "race_bin", "age_bin"]]
        raise ValueError(
            "Missing ACS targets for some calibration cells. "
            "Check bin labels and that ACS table columns match (race_bin, age_bin).\n"
            f"Example missing cells:\n{miss.head(10)}"
        )

    cells["pop_total_y"] = cells.groupby(year_col)["n_pop"].transform("sum")
    cells["sample_total_y"] = cells.groupby(year_col)["n_sample"].transform("sum")
    cells["w_raw"] = (cells["n_pop"] / cells["pop_total_y"]) * (cells["sample_total_y"] / cells["n_sample"])
    
    df = df.merge(
        cells[[year_col, "geo_bin", "race_bin", "age_bin", "w_raw"]],
        on=[year_col, "geo_bin", "race_bin", "age_bin"],
        how="left"
    )

    df["w_in_calibration"] = df["w_raw"].notna().astype(int)
    df["w_used"] = df["w_raw"].fillna(1.0)
    df["w_trim"] = df["w_used"].astype(float)
    
    for y in year_levels:
        mask = (df[year_col] == y) & (df["w_in_calibration"] == 1)
        if mask.sum() == 0:
            continue
        lo, hi = df.loc[mask, "w_used"].quantile([trim_q[0], trim_q[1]]).tolist()
        df.loc[df[year_col] == y, "w_trim"] = df.loc[df[year_col] == y, "w_trim"].clip(lower=lo, upper=hi)
    if max_cap is not None:
        df["w_trim"] = df["w_trim"].clip(upper=float(max_cap))
    return df

def weight_diagnostics(df_w: pd.DataFrame, year_col="SurveyYear", w_col="w_trim") -> pd.DataFrame:
    rows = []
    for y, g in df_w.groupby(year_col):
        w = g[w_col].astype(float)
        ess = (w.sum() ** 2) / (w.pow(2).sum()) if w.pow(2).sum() > 0 else np.nan
        rows.append({
            "SurveyYear": y,
            "N": len(g),
            "share_in_calibration": g["w_in_calibration"].mean() if "w_in_calibration" in g else np.nan,
            "sum_w": w.sum(),
            "mean_w": w.mean(),
            "p01": w.quantile(0.01),
            "p05": w.quantile(0.05),
            "p95": w.quantile(0.95),
            "p99": w.quantile(0.99),
            "min_w": w.min(),
            "max_w": w.max(),
            "ESS": ess
        })
    return pd.DataFrame(rows)

BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
STATE_FIPS = "24"  # Maryland

RACE_SUFFIXES = {
    "non-hispanic white": "H",   # White Alone, Not Hispanic or Latino
    "non-hispanic black": "B",   # Black or African American Alone
    "hispanic": "I",             # Hispanic or Latino
    "other": {              
        "C": "American Indian and Alaska Native Alone",
        "D": "Asian Alone",
        "E": "Native Hawaiian and Other Pacific Islander Alone",
        "F": "Some Other Race Alone",
        "G": "Two or More Races",
    },
}

# B03002: Hispanic or Latino Origin by Race (single table, no suffix)
B03002_FIELDS = {
    "non-hispanic white": ["B03002_003E"],
    "non-hispanic black": ["B03002_004E"],
    "hispanic": ["B03002_012E"],
    "other": [
        "B03002_005E",  # AIAN Alone
        "B03002_006E",  # Asian Alone
        "B03002_007E",  # NHPI Alone
        "B03002_008E",  # Some Other Race Alone
        "B03002_009E",  # Two or More Races
    ],
}

# B01001{suffix}: Sex by Age — fields for age brackets
# Suffix is appended per race group
B01001_FIELDS = {
    "total": "{prefix}_001E",
    "0-17": [
        "{prefix}_003E",  # Male <5
        "{prefix}_004E",  # Male 5-9
        "{prefix}_005E",  # Male 10-14
        "{prefix}_006E",  # Male 15-17
        "{prefix}_018E",  # Female <5
        "{prefix}_019E",  # Female 5-9
        "{prefix}_020E",  # Female 10-14
        "{prefix}_021E",  # Female 15-17
    ],
    "18-34": [
        "{prefix}_007E",  # Male 18-19
        "{prefix}_008E",  # Male 20-24
        "{prefix}_009E",  # Male 25-29
        "{prefix}_010E",  # Male 30-34
        "{prefix}_022E",  # Female 18-19
        "{prefix}_023E",  # Female 20-24
        "{prefix}_024E",  # Female 25-29
        "{prefix}_025E",  # Female 30-34
    ],
    "35-54": [
        "{prefix}_011E",  # Male 35-44
        "{prefix}_012E",  # Male 45-54
        "{prefix}_026E",  # Female 35-44
        "{prefix}_027E",  # Female 45-54
    ],
    "55+": [
        "{prefix}_013E",  # Male 55-64
        "{prefix}_014E",  # Male 65-74
        "{prefix}_015E",  # Male 75-84
        "{prefix}_016E",  # Male 85+
        "{prefix}_028E",  # Female 55-64
        "{prefix}_029E",  # Female 65-74
        "{prefix}_030E",  # Female 75-84
        "{prefix}_031E",  # Female 85+
    ],
}

# C23002{suffix}: Sex by Age by Employment Status
C23002_FIELDS = {
    "total": "{prefix}_001E",
    "employed": [
        "{prefix}_005E",  # Male 16-64 In Armed Forces
        "{prefix}_007E",  # Male 16-64 Employed
        "{prefix}_012E",  # Male 65+ Employed
        "{prefix}_018E",  # Female 16-64 In Armed Forces
        "{prefix}_020E",  # Female 16-64 Employed
        "{prefix}_025E",  # Female 65+ Employed
    ],
}


def _build_url(year: int) -> str:
    return BASE_URL.format(year=year)


def _api_get(
    year: int,
    fields: list[str],
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Query Census API for Maryland counties.
    """
    url = _build_url(year)

    MAX_FIELDS_PER_CALL = 47

    all_data = None

    for i in range(0, len(fields), MAX_FIELDS_PER_CALL):
        chunk = fields[i : i + MAX_FIELDS_PER_CALL]
        get_str = ",".join(["NAME"] + chunk)

        params = {
            "get": get_str,
            "for": "county:*",
            "in": f"state:{STATE_FIPS}",
        }
        if api_key:
            params["key"] = api_key

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  API retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Census API failed after {max_retries} attempts: {e}\n"
                        f"URL: {url}\nParams: {params}"
                    ) from e

        data = resp.json()
        header = data[0]
        rows = data[1:]

        df_chunk = pd.DataFrame(rows, columns=header)
        df_chunk = df_chunk.set_index("NAME")

        geo_cols = [c for c in df_chunk.columns if c in ("state", "county")]
        if all_data is None:
            geo_info = df_chunk[geo_cols].copy() if geo_cols else None
            df_chunk = df_chunk.drop(columns=geo_cols, errors="ignore")
            all_data = df_chunk
        else:
            df_chunk = df_chunk.drop(columns=geo_cols, errors="ignore")
            all_data = all_data.join(df_chunk, how="outer")

        time.sleep(0.5)

    for c in all_data.columns:
        all_data[c] = pd.to_numeric(all_data[c], errors="coerce")

    all_data.index = (
        all_data.index.str.replace(", Maryland", "", regex=False)
        .str.replace(" County", "", regex=False)
        .str.replace(" city", " City", regex=False)
    )

    return all_data


def _resolve_fields(template: str | list[str], prefix: str) -> list[str] | str:
    """Replace {prefix} placeholder in field templates."""
    if isinstance(template, str):
        return template.format(prefix=prefix)
    return [f.format(prefix=prefix) for f in template]


def _get_suffixes_for_race(race_group: str) -> list[str]:
    """Get the Census table suffixes for a race/ethnicity group."""
    suffix_info = RACE_SUFFIXES[race_group]
    if isinstance(suffix_info, str):
        return [suffix_info]
    elif isinstance(suffix_info, dict):
        return list(suffix_info.keys())
    else:
        raise ValueError(f"Unexpected suffix type for {race_group}")


def fetch_population_by_race(
    year: int = 2024,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch county population by race/ethnicity from table B03002.
    """
    print(f"Fetching B03002 (population by race/ethnicity) for {year}...")
    all_fields = []
    for group_fields in B03002_FIELDS.values():
        all_fields.extend(group_fields)

    df = _api_get(year, all_fields, api_key=api_key)

    result = {}
    for group, group_fields in B03002_FIELDS.items():
        result[group] = df[group_fields].sum(axis=1)

    return pd.DataFrame(result)


def fetch_age_by_race(
    year: int = 2024,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch age distribution by race/ethnicity from B01001{suffix} tables.
    """
    print(f"Fetching B01001 variants (age by race) for {year}...")

    age_groups = ["0-17", "18-34", "35-54", "55+"]
    results = []

    for race_group in ["non-hispanic white", "non-hispanic black", "hispanic", "other"]:
        suffixes = _get_suffixes_for_race(race_group)

        for age_group in age_groups:
            group_totals = {}
            group_counts = {}

            for suffix in suffixes:
                prefix = f"B01001{suffix}"

                total_field = _resolve_fields(B01001_FIELDS["total"], prefix)
                count_fields = _resolve_fields(B01001_FIELDS[age_group], prefix)
                all_fields_for_suffix = [total_field] + count_fields

                df = _api_get(year, all_fields_for_suffix, api_key=api_key)

                group_totals[suffix] = df[total_field]
                group_counts[suffix] = df[count_fields].sum(axis=1)

            total_series = pd.DataFrame(group_totals).sum(axis=1)
            count_series = pd.DataFrame(group_counts).sum(axis=1)
            pct_series = count_series / total_series

            results.append(count_series.rename(f"{race_group}_{age_group}_cnt"))
            results.append(total_series.rename(f"{race_group}_{age_group}_tot"))
            results.append(pct_series.rename(f"{race_group}_{age_group}_pct"))

    result_df = pd.DataFrame(results).T
    result_df.columns = result_df.columns.str.split("_", n=2, expand=True)

    return result_df


def fetch_employment_by_race(
    year: int = 2024,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch employment rates by race/ethnicity from C23002{suffix} tables.
    """
    print(f"Fetching C23002 variants (employment by race) for {year}...")

    results = []

    for race_group in ["non-hispanic white", "non-hispanic black", "hispanic", "other"]:
        suffixes = _get_suffixes_for_race(race_group)

        for emp_group in ["employed"]:
            group_totals = {}
            group_counts = {}

            for suffix in suffixes:
                prefix = f"C23002{suffix}"

                total_field = _resolve_fields(C23002_FIELDS["total"], prefix)
                count_fields = _resolve_fields(C23002_FIELDS[emp_group], prefix)
                all_fields_for_suffix = [total_field] + count_fields

                df = _api_get(year, all_fields_for_suffix, api_key=api_key)

                group_totals[suffix] = df[total_field]
                group_counts[suffix] = df[count_fields].sum(axis=1)

            total_series = pd.DataFrame(group_totals).sum(axis=1)
            count_series = pd.DataFrame(group_counts).sum(axis=1)
            pct_series = count_series / total_series

            results.append(count_series.rename(f"{race_group}_{emp_group}_cnt"))
            results.append(total_series.rename(f"{race_group}_{emp_group}_tot"))
            results.append(pct_series.rename(f"{race_group}_{emp_group}_pct"))

    result_df = pd.DataFrame(results).T
    result_df.columns = result_df.columns.str.split("_", n=2, expand=True)

    return result_df

     
def county_employed_population(
    year: int = 2024,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Estimate employed population by county, race/ethnicity, and age bracket.
    """
    print(f"\n{'='*60}")
    print(f"Building county employed population table")
    print(f"ACS 5-Year ending: {year}  (covers {year-4}-{year})")
    print(f"{'='*60}\n")

    county_pops = fetch_population_by_race(year, api_key)
    print(f"  Counties retrieved: {len(county_pops)}")

    county_ages = fetch_age_by_race(year, api_key)

    county_emp = fetch_employment_by_race(year, api_key)

    # Compute employed population by (county, race, age)

    adult_ages = ["18-34", "35-54", "55+"]
    race_groups = ["non-hispanic white", "non-hispanic black", "hispanic", "other"]

    result_cols = pd.MultiIndex.from_product(
        [race_groups, adult_ages], names=["race_bin", "age_bin"]
    )
    final = pd.DataFrame(index=county_pops.index, columns=result_cols, dtype=float)

    for race in race_groups:
        total_pop_col = (race, "0-17", "tot")
        under18_col = (race, "0-17", "cnt")

        if total_pop_col in county_ages.columns and under18_col in county_ages.columns:
            pop_over_18 = county_ages[total_pop_col] - county_ages[under18_col]
        else:
            pop_over_18 = county_pops[race]

        emp_cnt_col = (race, "employed", "cnt")
        emp_tot_col = (race, "employed", "tot")

        pct_employed = county_emp[emp_cnt_col] / pop_over_18
        pct_employed = pct_employed.clip(upper=1.0)  

        for age in adult_ages:
            pct_col = (race, age, "pct")
            if pct_col in county_ages.columns:
                age_pct = county_ages[pct_col]
            else:
                age_pct = pd.Series(1.0 / len(adult_ages), index=county_pops.index)

            final[(race, age)] = county_pops[race] * age_pct * pct_employed

    final = final.round(0)

    total_employed = final.sum().sum()
    print(f"\n  Total estimated employed population: {total_employed:,.0f}")
    print(f"  Counties: {len(final)}")
    print(f"  Columns: {list(final.columns.get_level_values(0).unique())}")
    print(f"  Age bins: {list(final.columns.get_level_values(1).unique())}")

    return final


def summarize_weight_cells(
    df_w,
    year_col="SurveyYear",
    geo_col="geo_bin",
    race_col="race_bin",
    age_col="age_bin",
    weight_col="w_trim",
    pop_col="n_pop",
    sample_col="n_sample",
    top_k=15,
):
    """
      - n_sample, share_sample (within year)
      - n_pop,    share_pop    (within year)
      - ratio = share_pop/share_sample
      - weight (mean within cell)
    """
    keys = [year_col, geo_col, race_col, age_col]
    sample_cells = (
        df_w.dropna(subset=keys)
            .groupby(keys)
            .size()
            .rename("n_sample")
            .reset_index()
    )
    w_cells = (
        df_w.dropna(subset=keys + [weight_col])
            .groupby(keys)[weight_col]
            .agg(weight_mean="mean", weight_min="min", weight_max="max")
            .reset_index()
    )
    cells = sample_cells.merge(w_cells, on=keys, how="left")
    if pop_col in df_w.columns:
        pop_cells = (
            df_w.dropna(subset=keys + [pop_col])
                .groupby(keys)[pop_col]
                .first() 
                .rename("n_pop")
                .reset_index()
        )
        cells = cells.merge(pop_cells, on=keys, how="left")

    cells["sample_total_y"] = cells.groupby(year_col)["n_sample"].transform("sum")
    cells["share_sample"] = cells["n_sample"] / cells["sample_total_y"]

    if "n_pop" in cells.columns and cells["n_pop"].notna().any():
        cells["pop_total_y"] = cells.groupby(year_col)["n_pop"].transform("sum")
        cells["share_pop"] = cells["n_pop"] / cells["pop_total_y"]
        cells["ratio_share_pop_to_sample"] = cells["share_pop"] / cells["share_sample"]
    else:
        cells["share_pop"] = np.nan
        cells["ratio_share_pop_to_sample"] = np.nan

    out_cols = keys + [
        "n_sample", "share_sample",
        "n_pop", "share_pop",
        "ratio_share_pop_to_sample",
        "weight_mean", "weight_min", "weight_max"
    ]
    out_cols = [c for c in out_cols if c in cells.columns]
    cells_full = cells[out_cols].sort_values(keys)
    top_by_weight = cells_full.sort_values("weight_mean", ascending=False).head(top_k)
    if "ratio_share_pop_to_sample" in cells_full.columns and cells_full["ratio_share_pop_to_sample"].notna().any():
        top_by_ratio = cells_full.sort_values("ratio_share_pop_to_sample", ascending=False).head(top_k)
    else:
        top_by_ratio = None
    return cells_full, top_by_weight, top_by_ratio

def build_reviewproof_cell_table(df_m_w, pop_targets, top_k=15):
    keys = ["SurveyYear", "geo_bin", "race_bin", "age_bin"]
    df_tmp = df_m_w.merge(pop_targets[keys + ["n_pop"]], on=keys, how="left")
    return summarize_weight_cells(df_tmp, top_k=top_k)

def make_pop_targets_from_county_employed_tbl(county_employed_tbl, df_m_w, county_col="home_county_clean"):
    """
    Build ACS pop targets for (geo_bin, race_bin, age_bin) and replicate for each SurveyYear in df_m_w.
    """
    acs = county_employed_tbl.copy()
    acs.index = acs.index.map(clean_md_county_name)
    acs = acs.groupby(level=0).sum() 
    
    tmp = df_m_w[[county_col, "geo_bin"]].dropna().copy()
    tmp[county_col] = tmp[county_col].map(clean_md_county_name)
    county_geo = (
        tmp.groupby(county_col)["geo_bin"]
           .agg(lambda s: s.value_counts().idxmax())
           .rename("geo_bin")
           .reset_index()
           .rename(columns={county_col: "county"})
    )
    pop_long = acs.stack([0, 1]).rename("n_pop").reset_index()
    pop_long.columns = ["county", "race_bin", "age_bin", "n_pop"]
    pop_long["county"] = pop_long["county"].map(clean_md_county_name)
    pop_long = pop_long.merge(county_geo, on="county", how="left")
    pop_long["geo_bin"] = pop_long["geo_bin"].fillna("Outlying")  
    pop_targets_core = (
        pop_long.groupby(["geo_bin", "race_bin", "age_bin"], as_index=False)["n_pop"].sum()
    )
    years = sorted(df_m_w["SurveyYear"].dropna().unique().tolist())
    pop_targets = (
        pop_targets_core.assign(_k=1)
        .merge(pd.DataFrame({"SurveyYear": years}).assign(_k=1), on="_k")
        .drop(columns="_k")
    )
    return pop_targets
