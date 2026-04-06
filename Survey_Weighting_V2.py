# Maryland Post-Stratification Weights (Iterative Raking)

# Calibration margins:
#   Margin 1: SurveyYear × geo_bin × race_bin × age_bin
#   Margin 2: SurveyYear × edu_bin
#
# geo_bin:  Central / Outlying
# race_bin: non-hispanic white / non-hispanic black / hispanic / other
# age_bin:  18-34 / 35-54 / 55+
# edu_bin:  pre_bachelor / bachelor_degree / graduate_degree

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

def _norm_text(s: str) -> str:
    return str(s).strip() if not pd.isna(s) else np.nan


def clean_md_county_name(x: str) -> str:
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace(" County", "").replace(" county", "")
    low = x.lower()

    if low in ("baltimore city", "baltimore city.", "baltimorecity"):
        return "Baltimore City"
    if low in (
        "saint mary's", "saint marys", "saint mary's city", "saint mary",
        "st. mary's", "st mary's", "st marys"
    ):
        return "St. Mary's"
    if low in ("prince george's", "prince georges", "prince george’s"):
        return "Prince George's"

    x = x.replace("CITY", "City").replace("city", "City")
    return x


def map_geo_bin(x: str) -> str:
    if pd.isna(x):
        return np.nan
    x = _norm_text(x)
    if x == "Central":
        return "Central"
    if x in ("Outlying", "Outside_CBSA_or_Unclassified"):
        return "Outlying"
    return np.nan


def map_age_bin(age_clean: pd.Series) -> pd.Series:
    return age_clean.map({
        "18_34": "18-34",
        "35_54": "35-54",
        "55_plus": "55+"
    })


def map_race_bin(race_clean: pd.Series) -> pd.Series:
    mapping = {
        "non_hispanic_white": "non-hispanic white",
        "non_hispanic_black": "non-hispanic black",
        "hispanic": "hispanic",
        "other_multiracial": "other",
    }

    def _f(x):
        if pd.isna(x):
            return np.nan
        x = _norm_text(x)
        return mapping.get(x, "other" if x.lower() != "unknown" else np.nan)

    return race_clean.apply(_f)


def map_edu_bin(edu_clean: pd.Series) -> pd.Series:
    mapping = {
        "pre_bachelor": "pre_bachelor",
        "bachelor_degree": "bachelor_degree",
        "graduate_degree": "graduate_degree",
    }
    return edu_clean.map(mapping)


BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
STATE_FIPS = "24"  # Maryland


def _api_get(
    year: int,
    fields: list[str],
    api_key: Optional[str] = None,
    max_retries: int = 3
) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    max_per_call = 47
    all_data = None

    for i in range(0, len(fields), max_per_call):
        chunk = fields[i:i + max_per_call]
        params = {
            "get": ",".join(["NAME"] + chunk),
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
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(f"Census API failed: {e}") from e

        data = resp.json()
        df_chunk = pd.DataFrame(data[1:], columns=data[0]).set_index("NAME")
        geo_cols = [c for c in df_chunk.columns if c in ("state", "county")]

        if all_data is None:
            all_data = df_chunk.drop(columns=geo_cols, errors="ignore")
        else:
            all_data = all_data.join(
                df_chunk.drop(columns=geo_cols, errors="ignore"),
                how="outer",
            )

        time.sleep(0.5)

    for c in all_data.columns:
        all_data[c] = pd.to_numeric(all_data[c], errors="coerce")

    all_data.index = (
        all_data.index.str.replace(", Maryland", "", regex=False)
        .str.replace(" County", "", regex=False)
        .str.replace(" city", " City", regex=False)
    )
    return all_data

RACE_SUFFIXES = {
    "non-hispanic white": ["H"],
    "non-hispanic black": ["B"],
    "hispanic": ["I"],
    "other": ["C", "D", "E", "F", "G"],
}

# B03002: Hispanic or Latino Origin by Race
B03002_FIELDS = {
    "non-hispanic white": ["B03002_003E"],
    "non-hispanic black": ["B03002_004E"],
    "hispanic": ["B03002_012E"],
    "other": [
        "B03002_005E", "B03002_006E", "B03002_007E",
        "B03002_008E", "B03002_009E"
    ],
}

# B01001{suffix}: Sex by Age
B01001_AGE_FIELDS = {
    "0-17": [
        "{p}_003E", "{p}_004E", "{p}_005E", "{p}_006E",
        "{p}_018E", "{p}_019E", "{p}_020E", "{p}_021E"
    ],
    "18-34": [
        "{p}_007E", "{p}_008E", "{p}_009E", "{p}_010E",
        "{p}_022E", "{p}_023E", "{p}_024E", "{p}_025E"
    ],
    "35-54": ["{p}_011E", "{p}_012E", "{p}_026E", "{p}_027E"],
    "55+": [
        "{p}_013E", "{p}_014E", "{p}_015E", "{p}_016E",
        "{p}_028E", "{p}_029E", "{p}_030E", "{p}_031E"
    ],
}

# C23002{suffix}: Sex by Age by Employment Status
C23002_EMP_FIELDS = [
    "{p}_005E", "{p}_007E", "{p}_012E",
    "{p}_018E", "{p}_020E", "{p}_025E",
]

# B15002: Sex by Educational Attainment (25+, all races, detailed)
B15002_EDU_MAP = {
    "pre_bachelor": [
        "B15002_003E", "B15002_004E", "B15002_005E", "B15002_006E",
        "B15002_007E", "B15002_008E", "B15002_009E", "B15002_010E",
        "B15002_011E",
        "B15002_020E", "B15002_021E", "B15002_022E", "B15002_023E",
        "B15002_024E", "B15002_025E", "B15002_026E", "B15002_027E",
        "B15002_028E",
    ],
    "bachelor_degree": ["B15002_012E", "B15002_029E"],
    "graduate_degree": [
        "B15002_013E", "B15002_014E", "B15002_015E",
        "B15002_030E", "B15002_031E", "B15002_032E",
    ],
}

def _fmt(templates, prefix):
    if isinstance(templates, str):
        return templates.format(p=prefix)
    return [t.format(p=prefix) for t in templates]


def fetch_population_by_race(
    year: int,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    all_fields = [f for flds in B03002_FIELDS.values() for f in flds]
    df = _api_get(year, all_fields, api_key)
    return pd.DataFrame({
        grp: df[flds].sum(axis=1) for grp, flds in B03002_FIELDS.items()
    })


def fetch_age_by_race(
    year: int,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    results = {}
    for race, suffixes in RACE_SUFFIXES.items():
        for age in ["0-17", "18-34", "35-54", "55+"]:
            totals, counts = [], []
            for sfx in suffixes:
                prefix = f"B01001{sfx}"
                total_f = f"{prefix}_001E"
                count_fs = _fmt(B01001_AGE_FIELDS[age], prefix)
                df = _api_get(year, [total_f] + count_fs, api_key)
                totals.append(df[total_f])
                counts.append(df[count_fs].sum(axis=1))

            total_s = pd.concat(totals, axis=1).sum(axis=1)
            count_s = pd.concat(counts, axis=1).sum(axis=1)

            results[(race, age, "cnt")] = count_s
            results[(race, age, "tot")] = total_s
            results[(race, age, "pct")] = count_s / total_s

    out = pd.DataFrame(results)
    out.columns = pd.MultiIndex.from_tuples(
        out.columns,
        names=["race_bin", "age_bin", "stat"]
    )
    return out


def fetch_employment_by_race(
    year: int,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    results = {}
    for race, suffixes in RACE_SUFFIXES.items():
        totals, counts = [], []
        for sfx in suffixes:
            prefix = f"C23002{sfx}"
            total_f = f"{prefix}_001E"
            emp_fs = _fmt(C23002_EMP_FIELDS, prefix)
            df = _api_get(year, [total_f] + emp_fs, api_key)
            totals.append(df[total_f])
            counts.append(df[emp_fs].sum(axis=1))

        results[(race, "employed", "cnt")] = pd.concat(counts, axis=1).sum(axis=1)
        results[(race, "employed", "tot")] = pd.concat(totals, axis=1).sum(axis=1)

    out = pd.DataFrame(results)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def fetch_education_by_county(
    year: int,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch B15002 (education for 25+ population) by county.
    Returns columns:
      pre_bachelor, bachelor_degree, graduate_degree
    """
    all_fields = [f for flds in B15002_EDU_MAP.values() for f in flds]
    df = _api_get(year, all_fields, api_key)
    return pd.DataFrame({
        edu: df[flds].sum(axis=1) for edu, flds in B15002_EDU_MAP.items()
    })

def county_employed_population(
    year: int = 2024,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Estimate employed population by county × race × age.
    Returns a DataFrame indexed by county with MultiIndex columns
    (race_bin, age_bin).
    """
    print(f"Building county employed population table (ACS 5-Year ending {year})")

    pops = fetch_population_by_race(year, api_key)
    ages = fetch_age_by_race(year, api_key)
    emps = fetch_employment_by_race(year, api_key)

    adult_ages = ["18-34", "35-54", "55+"]
    race_groups = list(RACE_SUFFIXES.keys())

    cols = pd.MultiIndex.from_product(
        [race_groups, adult_ages],
        names=["race_bin", "age_bin"]
    )
    final = pd.DataFrame(index=pops.index, columns=cols, dtype=float)

    for race in race_groups:
        pop_over18 = ages[(race, "0-17", "tot")] - ages[(race, "0-17", "cnt")]
        pct_emp = (emps[(race, "employed", "cnt")] / pop_over18).clip(upper=1.0)

        for age in adult_ages:
            age_pct = ages[(race, age, "pct")]
            final[(race, age)] = pops[race] * age_pct * pct_emp

    final = final.round(0)
    print(f"  Total estimated employed: {final.sum().sum():,.0f}")
    print(f"  Counties: {len(final)}")
    return final


def build_edu_targets(
    county_edu_tbl: pd.DataFrame,
    county_to_geo: pd.Series,
    target_counties: list,
    year_levels: list
) -> pd.DataFrame:
    """
    Build education margin targets for edu_bin only.
    Uses B15002 county-level data for the 25+ population, aggregated over
    target counties.
    """
    tbl = county_edu_tbl.copy()
    tbl.index = tbl.index.map(clean_md_county_name)
    target_counties = [clean_md_county_name(c) for c in target_counties]
    tbl = tbl.loc[tbl.index.isin(target_counties)]

    # county_to_geo retained as an argument for symmetry/future extension
    _ = county_to_geo

    edu_totals = tbl[["pre_bachelor", "bachelor_degree", "graduate_degree"]].sum()
    edu_share = edu_totals / edu_totals.sum()

    rows = []
    for y in year_levels:
        for edu, share in edu_share.items():
            rows.append({
                "SurveyYear": str(y),
                "edu_bin": edu,
                "edu_share": share
            })

    return pd.DataFrame(rows)


def build_demo_targets(
    county_employed_tbl: pd.DataFrame,
    county_to_geo: pd.Series,
    target_counties: list,
    year_levels: list
) -> pd.DataFrame:
    """
    Build demographic margin targets: geo_bin × race_bin × age_bin.
    """
    tbl = county_employed_tbl.copy()
    tbl.index = tbl.index.map(clean_md_county_name)
    target_counties = [clean_md_county_name(c) for c in target_counties]
    tbl = tbl.loc[tbl.index.isin(target_counties)]

    geo_map = county_to_geo.copy()
    geo_map.index = geo_map.index.map(clean_md_county_name)

    long = (
        tbl.stack([0, 1])
        .rename("n_pop")
        .reset_index()
    )
    long.columns = ["county_clean", "race_bin", "age_bin", "n_pop"]

    long["geo_bin"] = long["county_clean"].map(geo_map).fillna("Outlying")
    long["n_pop"] = pd.to_numeric(long["n_pop"], errors="coerce")

    if long["n_pop"].isna().any():
        raise ValueError(
            "Non-numeric values found while building demographic targets:\n"
            f"{long.loc[long['n_pop'].isna()].head(10)}"
        )

    long = (
        long.groupby(["geo_bin", "race_bin", "age_bin"], as_index=False)["n_pop"]
        .sum()
    )

    return pd.concat(
        [long.assign(SurveyYear=str(y)) for y in year_levels],
        ignore_index=True,
    )

def unsupported_target_cells(
    df_year: pd.DataFrame,
    targets: pd.DataFrame,
    margin_keys: list
) -> pd.DataFrame:
    """
    Return target cells that are absent from the sample support for a given year.
    """
    sample_cells = df_year[margin_keys].drop_duplicates()
    out = targets.merge(sample_cells, on=margin_keys, how="left", indicator=True)
    out = out.loc[out["_merge"] == "left_only"].drop(columns="_merge")
    return out


def _rake_one_margin(
    df: pd.DataFrame,
    w_col: str,
    margin_keys: list,
    target_col: str,
    targets: pd.DataFrame,
    renormalize_targets: bool = True
) -> pd.DataFrame:
    """
    Single raking pass.
    """
    df = df.copy()

    cell_sums = (
        df.groupby(margin_keys, dropna=False)[w_col]
        .sum()
        .rename("w_sum")
        .reset_index()
    )

    cell_sums = cell_sums.merge(targets, on=margin_keys, how="left")

    if cell_sums[target_col].isna().any():
        missing_target_cells = cell_sums.loc[cell_sums[target_col].isna(), margin_keys]
        raise ValueError(
            f"Missing target shares for some observed sample cells:\n"
            f"{missing_target_cells.head(10)}"
        )

    if renormalize_targets:
        tgt_sum = cell_sums[target_col].sum()
        if tgt_sum <= 0 or pd.isna(tgt_sum):
            raise ValueError(
                f"Target shares sum to {tgt_sum} on observed support for margin {margin_keys}"
            )
        cell_sums["target_share_adj"] = cell_sums[target_col] / tgt_sum
    else:
        cell_sums["target_share_adj"] = cell_sums[target_col]

    total_w = df[w_col].sum()
    cell_sums["target_w"] = cell_sums["target_share_adj"] * total_w
    cell_sums["adj_factor"] = cell_sums["target_w"] / cell_sums["w_sum"]
    cell_sums["adj_factor"] = (
        cell_sums["adj_factor"]
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    df = df.merge(
        cell_sums[margin_keys + ["adj_factor"]],
        on=margin_keys,
        how="left",
    )
    df[w_col] = df[w_col] * df["adj_factor"].fillna(1.0)
    df.drop(columns=["adj_factor"], inplace=True)
    return df


def iterative_rake(
    df: pd.DataFrame,
    w_col: str,
    margins: list[dict],
    max_iter: int = 150,
    tol: float = 1e-3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Iterative proportional fitting across multiple margins.
    """
    df = df.copy()

    for iteration in range(max_iter):
        w_before = df[w_col].copy()

        for margin in margins:
            df = _rake_one_margin(
                df=df,
                w_col=w_col,
                margin_keys=margin["keys"],
                target_col=margin["target_col"],
                targets=margin["targets"],
                renormalize_targets=True,
            )

        max_change = (df[w_col] - w_before).abs().max()
        mean_change = (df[w_col] - w_before).abs().mean()

        if max_change < tol:
            if verbose:
                print(
                    f"  Raking converged after {iteration + 1} iterations "
                    f"(max change: {max_change:.6f}, mean change: {mean_change:.6f})"
                )
            break
    else:
        if verbose:
            print(
                f"  Warning: raking did not converge after {max_iter} iterations "
                f"(max change: {max_change:.6f}, mean change: {mean_change:.6f})"
            )

    return df

def compute_md_poststrat_weights(
    df_model: pd.DataFrame,
    county_employed_tbl: pd.DataFrame,
    county_edu_tbl: pd.DataFrame,
    *,
    year_col: str = "SurveyYear",
    county_col: str = "home_county",
    geo_col: str = "home_central_outlying_county",
    race_col: str = "race_clean",
    age_col: str = "age_clean",
    edu_col: str = "education_clean",
    trim_q: tuple = (0.01, 0.99),
    max_cap: float = 10.0,
    target_universe: str = "model_counties",
    rake_max_iter: int = 150,
    rake_tol: float = 1e-3,
    normalize_trim_within_year: bool = True,
) -> pd.DataFrame:
    """
    Compute post-stratification weights using iterative raking on two margins:
      Margin 1: geo_bin × race_bin × age_bin
      Margin 2: edu_bin
    """
    df = df_model.copy()

    required_cols = [year_col, county_col, geo_col, race_col, age_col, edu_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in df_model: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["county_clean"] = df[county_col].map(clean_md_county_name)
    df["geo_bin"] = df[geo_col].map(map_geo_bin)
    df["race_bin"] = map_race_bin(df[race_col])
    df["age_bin"] = map_age_bin(df[age_col])
    df["edu_bin"] = map_edu_bin(df[edu_col])
    df[year_col] = df[year_col].astype(str)

    if target_universe == "model_counties":
        target_counties = sorted(df["county_clean"].dropna().unique())
    elif target_universe == "all_md_counties_in_acs":
        target_counties = sorted(
            pd.Index(county_employed_tbl.index)
            .map(clean_md_county_name)
            .dropna()
            .unique()
        )
    else:
        raise ValueError(
            "target_universe must be 'model_counties' or 'all_md_counties_in_acs'"
        )

    county_to_geo = (
        df.dropna(subset=["county_clean", "geo_bin"])
        .groupby("county_clean")["geo_bin"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Outlying")
    )

    year_levels = sorted(df[year_col].dropna().unique())

    # Build targets
    demo_targets = build_demo_targets(
        county_employed_tbl=county_employed_tbl,
        county_to_geo=county_to_geo,
        target_counties=target_counties,
        year_levels=year_levels,
    )
    demo_targets["n_pop"] = pd.to_numeric(demo_targets["n_pop"], errors="coerce")
    if demo_targets["n_pop"].isna().any():
        raise ValueError(
            "demo_targets['n_pop'] contains non-numeric values:\n"
            f"{demo_targets.loc[demo_targets['n_pop'].isna()].head(10)}"
        )

    demo_targets["pop_total_y"] = (
        demo_targets.groupby("SurveyYear")["n_pop"].transform("sum")
    )
    demo_targets["demo_share"] = demo_targets["n_pop"] / demo_targets["pop_total_y"]

    edu_targets = build_edu_targets(
        county_edu_tbl=county_edu_tbl,
        county_to_geo=county_to_geo,
        target_counties=target_counties,
        year_levels=year_levels,
    )

    df["w_in_calibration"] = (
        df[year_col].notna()
        & df["geo_bin"].notna()
        & df["race_bin"].notna()
        & df["age_bin"].notna()
        & df["edu_bin"].notna()
    ).astype(int)

    df["w_raw"] = 1.0
    year_dfs = []

    for y in year_levels:
        mask = (df[year_col] == y) & (df["w_in_calibration"] == 1)
        df_y = df.loc[mask].copy()

        if len(df_y) == 0:
            year_dfs.append(df.loc[df[year_col] == y].copy())
            continue

        demo_y = demo_targets.loc[demo_targets["SurveyYear"] == y].copy()
        edu_y = edu_targets.loc[edu_targets["SurveyYear"] == y].copy()

        unsupported_demo = unsupported_target_cells(
            df_year=df_y,
            targets=demo_y[["geo_bin", "race_bin", "age_bin", "demo_share"]],
            margin_keys=["geo_bin", "race_bin", "age_bin"],
        )
        unsupported_edu = unsupported_target_cells(
            df_year=df_y,
            targets=edu_y[["edu_bin", "edu_share"]],
            margin_keys=["edu_bin"],
        )

        margins = [
            {
                "keys": ["geo_bin", "race_bin", "age_bin"],
                "target_col": "demo_share",
                "targets": demo_y[["geo_bin", "race_bin", "age_bin", "demo_share"]],
            },
            {
                "keys": ["edu_bin"],
                "target_col": "edu_share",
                "targets": edu_y[["edu_bin", "edu_share"]],
            },
        ]

        print(f"\nRaking year {y} (n={len(df_y)})...")
        df_y = iterative_rake(
            df=df_y,
            w_col="w_raw",
            margins=margins,
            max_iter=rake_max_iter,
            tol=rake_tol,
            verbose=True,
        )

        df_y["w_raw"] = df_y["w_raw"] / df_y["w_raw"].mean()
        year_dfs.append(df_y)

        non_calib = df.loc[
            (df[year_col] == y) & (df["w_in_calibration"] == 0)
        ].copy()
        if len(non_calib) > 0:
            year_dfs.append(non_calib)

    df_out = pd.concat(year_dfs, ignore_index=True)

    df_out["w_trim"] = df_out["w_raw"].copy()

    for y in year_levels:
        mask = (df_out[year_col] == y) & (df_out["w_in_calibration"] == 1)
        if mask.sum() == 0:
            continue

        lo, hi = df_out.loc[mask, "w_raw"].quantile([trim_q[0], trim_q[1]])
        df_out.loc[mask, "w_trim"] = df_out.loc[mask, "w_trim"].clip(
            lower=lo,
            upper=hi,
        )

    if max_cap is not None:
        df_out["w_trim"] = df_out["w_trim"].clip(upper=max_cap)

    df_out.loc[df_out["w_in_calibration"] == 0, "w_trim"] = 1.0

    if normalize_trim_within_year:
        for y in year_levels:
            mask_y = df_out[year_col] == y
            if mask_y.sum() == 0:
                continue

            mean_w = df_out.loc[mask_y, "w_trim"].mean()
            if pd.notna(mean_w) and mean_w > 0:
                df_out.loc[mask_y, "w_trim"] = (
                    df_out.loc[mask_y, "w_trim"] / mean_w
                )

    return df_out

def weight_diagnostics(
    df_w: pd.DataFrame,
    year_col: str = "SurveyYear",
    w_col: str = "w_trim"
) -> pd.DataFrame:
    rows = []

    for y, g in df_w.groupby(year_col):
        w = g[w_col].astype(float)
        denom = w.pow(2).sum()
        ess = (w.sum() ** 2) / denom if denom > 0 else np.nan

        rows.append({
            "SurveyYear": y,
            "N": len(g),
            "share_in_calibration": g["w_in_calibration"].mean(),
            "sum_w": w.sum(),
            "mean_w": w.mean(),
            "min_w": w.min(),
            "p01": w.quantile(0.01),
            "p05": w.quantile(0.05),
            "median_w": w.median(),
            "p95": w.quantile(0.95),
            "p99": w.quantile(0.99),
            "max_w": w.max(),
            "ESS": ess,
            "ESS_pct": ess / len(g) * 100,
        })

    return pd.DataFrame(rows)


def margin_check(
    df_w: pd.DataFrame,
    margin_keys: list,
    targets: pd.DataFrame,
    target_col: str,
    w_col: str = "w_trim",
    year_col: str = "SurveyYear"
) -> pd.DataFrame:
    """
    Compare weighted sample margin shares against target shares.
    """
    check_rows = []

    for y, g in df_w.loc[df_w["w_in_calibration"] == 1].groupby(year_col):
        total_w = g[w_col].sum()

        cell_w = g.groupby(margin_keys)[w_col].sum().reset_index()
        cell_w["weighted_share"] = cell_w[w_col] / total_w
        cell_w[year_col] = y

        tgts = targets.loc[targets[year_col] == y, margin_keys + [target_col]]
        merged = cell_w.merge(tgts, on=margin_keys, how="outer")
        merged["diff"] = merged["weighted_share"] - merged[target_col]
        check_rows.append(merged)

    return pd.concat(check_rows, ignore_index=True)

def run_full_weighting_pipeline(
    df_model: pd.DataFrame,
    acs_year: int = 2024,
    api_key: Optional[str] = None,
    **weight_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline:
      1. Fetch ACS benchmarks
      2. Compute survey weights
      3. Return weighted data, diagnostics, and county employment table
    """
    county_emp_tbl = county_employed_population(acs_year, api_key)
    county_edu_tbl = fetch_education_by_county(acs_year, api_key)

    df_w = compute_md_poststrat_weights(
        df_model=df_model,
        county_employed_tbl=county_emp_tbl,
        county_edu_tbl=county_edu_tbl,
        **weight_kwargs,
    )

    diag = weight_diagnostics(df_w)
    diag.to_string(index=False)

    return df_w, diag, county_emp_tbl

def summarize_weighting_results(
    df_w: pd.DataFrame,
    demo_margin_check: pd.DataFrame,
    edu_margin_check: pd.DataFrame,
    unsupported_demo_by_year: dict | None = None,
    unsupported_edu_by_year: dict | None = None,
    year_col: str = "SurveyYear",
    w_col: str = "w_trim",
) -> pd.DataFrame:
    """
    Create an appendix-ready summary table by survey year.
    """
    unsupported_demo_by_year = unsupported_demo_by_year or {}
    unsupported_edu_by_year = unsupported_edu_by_year or {}

    diag = weight_diagnostics(df_w, year_col=year_col, w_col=w_col).copy()

    demo_mc = demo_margin_check.copy()
    demo_mc["abs_diff"] = demo_mc["diff"].abs()
    demo_summary = (
        demo_mc.groupby(year_col)["abs_diff"]
        .agg(demo_mean_abs_diff="mean", demo_max_abs_diff="max")
        .reset_index()
    )

    edu_mc = edu_margin_check.copy()
    edu_mc["abs_diff"] = edu_mc["diff"].abs()
    edu_summary = (
        edu_mc.groupby(year_col)["abs_diff"]
        .agg(edu_mean_abs_diff="mean", edu_max_abs_diff="max")
        .reset_index()
    )

    out = (
        diag.merge(demo_summary, on=year_col, how="left")
            .merge(edu_summary, on=year_col, how="left")
            .copy()
    )

    out["unsupported_demo_cells"] = out[year_col].astype(str).map(unsupported_demo_by_year).fillna(0).astype(int)
    out["unsupported_edu_cells"] = out[year_col].astype(str).map(unsupported_edu_by_year).fillna(0).astype(int)

    out["weight_design_effect"] = out["N"] / out["ESS"]

    cols = [
        year_col,
        "N",
        "share_in_calibration",
        "sum_w",
        "mean_w",
        "min_w",
        "median_w",
        "p95",
        "p99",
        "max_w",
        "ESS",
        "ESS_pct",
        "weight_design_effect",
        "unsupported_demo_cells",
        "unsupported_edu_cells",
        "demo_mean_abs_diff",
        "demo_max_abs_diff",
        "edu_mean_abs_diff",
        "edu_max_abs_diff",
    ]
    return out[cols].sort_values(year_col).reset_index(drop=True)

def collect_unsupported_cells_by_year(
    df_w: pd.DataFrame,
    county_emp_tbl: pd.DataFrame,
    county_edu_tbl: pd.DataFrame,
    year_col: str = "SurveyYear",
) -> tuple[dict, dict]:
    """
    Rebuild target structures and count unsupported target cells by year.
    """
    county_to_geo = (
        df_w.dropna(subset=["county_clean", "geo_bin"])
        .groupby("county_clean")["geo_bin"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Outlying")
    )

    year_levels = sorted(df_w[year_col].dropna().astype(str).unique())
    target_counties = sorted(df_w["county_clean"].dropna().unique())

    demo_targets = build_demo_targets(
        county_employed_tbl=county_emp_tbl,
        county_to_geo=county_to_geo,
        target_counties=target_counties,
        year_levels=year_levels,
    )
    demo_targets["pop_total_y"] = demo_targets.groupby("SurveyYear")["n_pop"].transform("sum")
    demo_targets["demo_share"] = demo_targets["n_pop"] / demo_targets["pop_total_y"]

    edu_targets = build_edu_targets(
        county_edu_tbl=county_edu_tbl,
        county_to_geo=county_to_geo,
        target_counties=target_counties,
        year_levels=year_levels,
    )

    unsupported_demo = {}
    unsupported_edu = {}

    for y in year_levels:
        df_y = df_w.loc[
            (df_w[year_col].astype(str) == str(y)) & (df_w["w_in_calibration"] == 1)
        ].copy()

        demo_y = demo_targets.loc[demo_targets["SurveyYear"] == str(y)].copy()
        edu_y = edu_targets.loc[edu_targets["SurveyYear"] == str(y)].copy()

        unsupported_demo[str(y)] = len(
            unsupported_target_cells(
                df_year=df_y,
                targets=demo_y[["geo_bin", "race_bin", "age_bin", "demo_share"]],
                margin_keys=["geo_bin", "race_bin", "age_bin"],
            )
        )
        unsupported_edu[str(y)] = len(
            unsupported_target_cells(
                df_year=df_y,
                targets=edu_y[["edu_bin", "edu_share"]],
                margin_keys=["edu_bin"],
            )
        )

    return unsupported_demo, unsupported_edu