import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from adjustText import adjust_text

import patsy
import statsmodels.api as sm
from scipy.stats import chi2, t

ARRANGEMENT_ORDER = ["In-Person", "Hybrid", "Remote"]

CATEGORY_ORDERS = {
    "arrangement": ["In-Person", "Hybrid", "Remote"],
    "teleworkable_label": ["Non-Teleworkable", "Teleworkable"],
    "age_display": ["18–34", "35–54", "55+"],
    "gender_display": ["Male", "Female"],
    "race_display": ["White non-Hispanic", "Black non-Hispanic", "Hispanic/Latinx", "Other/Multiracial"],
    "income_display": ["Under $50K", "$50K–$100K", "$100K–$150K", "Over $150K"],
    "lifecycle_display": ["Solitary", "Adults-Shared", "Family"],
    "vehicles_display": ["0 (car-free)", "1", "2", "3+"],
    "education_display": ["Pre-Bachelor", "Bachelor's degree", "Graduate degree"],
    "work_status_display": ["Full-time", "Part-time", "Other/Flexible"],
    "feasibility_display": ["None", "Partial", "Full"],
    "preference_display": ["No remote", "Some remote", "Full remote"],
    "geo_display": ["Central county", "Outlying county"],
    "home_cbsa_display": ["Baltimore metro", "Washington metro", "Other/Non-metro"],
    "spatial_mismatch_display": ["Same CBSA", "Cross-CBSA, same state", "Cross-state"],
    "ruca_home_display": ["Metro core", "Micro core", "Small town core", "Rural"],
    "ruca_work_display": ["Metro core", "Micro core", "Small town core", "Rural"],
    "year_display": ["2023", "2024", "2025"],

    # model-ready variables
    "teleworkable": ["Not Teleworkable", "Teleworkable"],
    "income": ["$50-100k", "<$50k", "$100-150k", ">$150k"],
    "age": ["35-54", "18-34", "55+"],
    "education": ["Bachelor's", "Pre-bachelor", "Graduate"],
    "employment": ["Full-time", "Not full-time"],
    "gender": ["Male", "Female"],
    "race": ["White", "Black", "Hispanic/Latinx", "Other"],
    "household": ["Adults-Shared", "Solitary", "Family"],
    "vehicles": ["0-1", "2+"],
    "county_type": ["Central", "Outlying"],
    "home_cbsa": ["Baltimore", "Washington", "Other"],
    "year": ["2023", "2024", "2025"],
    "major_soc_display": [
        "Management Occupations",
        "Office and Administrative Support Occupations",
        "Sales and Related Occupations",
        "Educational Instruction and Library Occupations",
        "Computer and Mathematical Occupations",
        "Food Preparation and Serving Related Occupations",
        "Business and Financial Operations Occupations",
        "Healthcare Practitioners and Technical Occupations",
        "Transportation and Material Moving Occupations",
        "Personal Care and Service Occupations",
        "Arts, Design, Entertainment, Sports, and Media Occupations",
        "Healthcare Support Occupations",
        "Construction and Extraction Occupations",
        "Protective Service Occupations",
        "Life, Physical, and Social Science Occupations",
        "Production Occupations",
        "Community and Social Service Occupations",
        "Architecture and Engineering Occupations",
        "Building and Grounds Cleaning and Maintenance Occupations",
        "Installation, Maintenance, and Repair Occupations",
        "Legal Occupations",
        "Farming, Fishing, and Forestry Occupations",
    ],
}

ONE_TO_ONE_MAPS = {
    "arrangement": (
        "work_remote_clean_",
        {
            "fully_in_person": "In-Person",
            "hybrid": "Hybrid",
            "fully_remote": "Remote",
        },
    ),
    "teleworkable_label": (
        "SOC_Teleworkable_Final",
        {1: "Teleworkable", 0: "Non-Teleworkable"},
    ),
    "education_display": (
        "education_clean",
        {
            "pre_bachelor": "Pre-Bachelor",
            "bachelor_degree": "Bachelor's degree",
            "graduate_degree": "Graduate degree",
        },
    ),
    "work_status_display": (
        "work_status_clean",
        {
            "full_time": "Full-time",
            "part_time": "Part-time",
            "other_flexible": "Other/Flexible",
        },
    ),
    "feasibility_display": (
        "remote_potential",
        {
            "full_remote_potential": "Full",
            "partial_remote_potential": "Partial",
            "no_remote_potential": "None",
        },
    ),
    "preference_display": (
        "remote_preference",
        {
            "full_remote_prefer": "Full remote",
            "partial_remote_prefer": "Some remote",
            "no_remote_prefer": "No remote",
        },
    ),
    "spatial_mismatch_display": (
        "hw_spatial_mismatch",
        {
            "same_cbsa": "Same CBSA",
            "cross_cbsa_same_state": "Cross-CBSA, same state",
            "cross_state": "Cross-state",
        },
    ),
    "ruca_home_display": (
        "ruca_home_UrbanCoreType",
        {
            "Metro core": "Metro core",
            "Micro core": "Micro core",
            "Small town core": "Small town core",
            "Rural": "Rural",
        },
    ),
    "ruca_work_display": (
        "ruca_work_UrbanCoreType",
        {
            "Metro core": "Metro core",
            "Micro core": "Micro core",
            "Small town core": "Small town core",
            "Rural": "Rural",
        },
    ),
}

GROUPED_RULES = {
    "age_display": (
        "age_clean",
        {
            "18–34": ["18_34"],
            "35–54": ["35_54"],
            "55+": ["55_plus"],
        },
    ),
    "gender_display": (
        "gender_clean",
        {
            "Male": ["male"],
            "Female": ["female"],
        },
    ),
    "race_display": (
        "race_clean",
        {
            "White non-Hispanic": ["non_hispanic_white"],
            "Black non-Hispanic": ["non_hispanic_black"],
            "Hispanic/Latinx": ["hispanic"],
            "Other/Multiracial": ["other_multiracial"],
        },
    ),
    "income_display": (
        "income_clean",
        {
            "Under $50K": ["under_50k"],
            "$50K–$100K": ["50k_100k"],
            "$100K–$150K": ["100k_150k"],
            "Over $150K": ["over_150k"],
        },
    ),
    "lifecycle_display": (
        "household_composition",
        {
            "Solitary": ["solitary"],
            "Adults-Shared": ["adults_shared"],
            "Family": ["family"],
        },
    ),
    "vehicles_display": (
        "hh_vehicles_clean",
        {
            "0 (car-free)": ["0_car_free"],
            "1": ["1_vehicle"],
            "2": ["2_vehicles"],
            "3+": ["3_plus_vehicles"],
        },
    ),
    "geo_display": (
        "home_central_outlying_county",
        {
            "Central county": ["Central"],
            "Outlying county": ["Outlying", "Non-central", "Other"],
        },
    ),
    "home_cbsa_display": (
        "home_cbsa_title",
        {
            "Baltimore metro": ["Baltimore-Columbia-Towson, MD"],
            "Washington metro": ["Washington-Arlington-Alexandria, DC-VA-MD-WV"],
            "Other/Non-metro": [],
        },
    ),
}

MODEL_MAPS = {
    "teleworkable": (
        "SOC_Teleworkable_Final",
        {1: "Teleworkable", 0: "Not Teleworkable"},
    ),
    "income": (
        "income_clean",
        {
            "under_50k": "<$50k",
            "50k_100k": "$50-100k",
            "100k_150k": "$100-150k",
            "over_150k": ">$150k",
        },
    ),
    "education": (
        "education_clean",
        {
            "pre_bachelor": "Pre-bachelor",
            "bachelor_degree": "Bachelor's",
            "graduate_degree": "Graduate",
        },
    ),
    "gender": (
        "gender_clean",
        {
            "male": "Male",
            "female": "Female",
        },
    ),
}

MODEL_GROUPED_RULES = {
    "age": (
        "age_clean",
        {
            "18-34": ["18_34"],
            "35-54": ["35_54"],
            "55+": ["55_plus"],
        },
    ),
    "employment": (
        "work_status_clean",
        {
            "Full-time": ["full_time"],
            "Not full-time": ["part_time", "other_flexible"],
        },
    ),
    "race": (
        "race_clean",
        {
            "White": ["non_hispanic_white"],
            "Black": ["non_hispanic_black"],
            "Hispanic/Latinx": ["hispanic"],
            "Other": ["other_multiracial"],
        },
    ),
    "household": (
        "household_composition",
        {
            "Adults-Shared": ["adults_shared"],
            "Solitary": ["solitary"],
            "Family": ["family"],
        },
    ),
    "vehicles": (
        "hh_vehicles_clean",
        {
            "0-1": ["0_car_free", "1_vehicle"],
            "2+": ["2_vehicles", "3_plus_vehicles"],
        },
    ),
    "county_type": (
        "home_central_outlying_county",
        {
            "Central": ["Central"],
            "Outlying": ["Outlying", "Non-central", "Other"],
        },
    ),
    "home_cbsa": (
        "home_cbsa_title",
        {
            "Baltimore": ["Baltimore-Columbia-Towson, MD"],
            "Washington": ["Washington-Arlington-Alexandria, DC-VA-MD-WV"],
            "Other": [],
        },
    ),
}

COLORS_ARRANGEMENT = {
    "In-Person": "#2C5F8A",
    "Hybrid": "#E8973A",
    "Remote": "#4A9B6E",
}

COLORS_TELEWORKABLE = {
    "Teleworkable": "#009E73",
    "Non-Teleworkable": "#D55E00",
}

COLORS_MODEL = {
    "Hybrid": "#E69F00",
    "Remote": "#0072B2",
}

MODEL_LABELS = {
    "teleworkableTeleworkable": "Teleworkable (Dingel-Neiman)",
    "industryProfessional/Finance/Information": "Professional/Finance/Info",
    "industryHealth Care": "Health Care",
    "industryEducation": "Education",
    "industryGovernment": "Government",
    "industryConstruction/Manufacturing/Transport": "Construction/Mfg/Transport",
    "income<$50k": "Income: <$50k",
    "income$100-150k": "Income: $100-150k",
    "income>$150k": "Income: >$150k",
    "educationPre-bachelor": "Pre-Bachelor",
    "educationGraduate": "Graduate Degree",
    "employmentNot full-time": "Not Full-Time",
    "age18-34": "Age: 18-34",
    "age55+": "Age: 55+",
    "genderFemale": "Female",
    "raceBlack": "Black",
    "raceHispanic/Latinx": "Hispanic/Latinx",
    "raceOther": "Other Race",
    "householdSolitary": "Solitary Household",
    "householdFamily": "Family Household",
    "vehicles2+": "2+ Vehicles",
    "county_typeOutlying": "Outlying County",
    "home_cbsaWashington": "Washington Metro",
    "home_cbsaOther": "Other/Non-Metro",
    "year2024": "Year: 2024",
    "year2025": "Year: 2025",
}

MODEL_TERM_ORDER = [
    "Teleworkable (Dingel-Neiman)",
    "Professional/Finance/Info",
    "Health Care",
    "Education",
    "Government",
    "Construction/Mfg/Transport",
    "Income: <$50k",
    "Income: $100-150k",
    "Income: >$150k",
    "Pre-Bachelor",
    "Graduate Degree",
    "Not Full-Time",
    "Age: 18-34",
    "Age: 55+",
    "Female",
    "Black",
    "Hispanic/Latinx",
    "Other Race",
    "Solitary Household",
    "Family Household",
    "2+ Vehicles",
    "Outlying County",
    "Washington Metro",
    "Other/Non-Metro",
    "Year: 2024",
    "Year: 2025",
]

def set_plot_style(base_size=15):
    plt.rcParams.update({
        "font.size": base_size - 1,
        "axes.titlesize": base_size + 2,
        "axes.titleweight": "bold",
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 2,
        "legend.title_fontsize": base_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })


def p_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def format_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.2f}"


def weighted_percent(series, weights, value):
    mask = series.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    s = series[mask]
    w = weights[mask]
    denom = w.sum()
    if denom == 0:
        return np.nan
    return 100 * w[s == value].sum() / denom


def weighted_chisq(df, row, col="arrangement", weight="w_trim"):
    """
    Weighted Pearson chi-square on weighted counts.
    This is an approximation to survey::svychisq, not a design-based Rao-Scott test.
    """
    sub = df[[row, col, weight]].dropna()
    if len(sub) < 10:
        return np.nan

    tab = pd.pivot_table(
        sub,
        values=weight,
        index=row,
        columns=col,
        aggfunc="sum",
        fill_value=0,
        observed=False
    )
    observed = tab.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return np.nan

    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / total

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((observed - expected) ** 2 / expected)

    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return 1 - chi2.cdf(stat, dof)


def recode_grouped(series, groups, default=np.nan):
    out = pd.Series(default, index=series.index, dtype="object")
    assigned = pd.Series(False, index=series.index)

    for new_value, raw_values in groups.items():
        if raw_values:
            mask = series.isin(raw_values)
            out.loc[mask] = new_value
            assigned.loc[mask] = True

    empty_keys = [k for k, v in groups.items() if len(v) == 0]
    if empty_keys:
        out.loc[~assigned & series.notna()] = empty_keys[0]

    return out


def apply_categorical_orders(df, category_orders):
    for col, levels in category_orders.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=levels, ordered=True)
    return df


def prepare_analysis_data(df):
    d = df.copy()

    for new_col, (src_col, mapping) in ONE_TO_ONE_MAPS.items():
        d[new_col] = d[src_col].map(mapping)

    for new_col, (src_col, groups) in GROUPED_RULES.items():
        d[new_col] = recode_grouped(d[src_col], groups)

    d["major_soc_display"] = d["Major_SOC_Occupations"]
    d["year_display"] = d["SurveyYear"].astype("Int64").astype(str)

    for new_col, (src_col, mapping) in MODEL_MAPS.items():
        d[new_col] = d[src_col].map(mapping)

    for new_col, (src_col, groups) in MODEL_GROUPED_RULES.items():
        d[new_col] = recode_grouped(d[src_col], groups)

    d["industry"] = d["NAICS_industry_collapsed"]
    d["industry"] = pd.Categorical(
        d["industry"],
        categories=[
            "Consumer-Facing Services",
            "Professional/Finance/Information",
            "Health Care",
            "Education",
            "Government",
            "Construction/Manufacturing/Transport",
            "Other Industries",
        ],
        ordered=True
    )
    d["year"] = d["SurveyYear"].astype("Int64").astype(str)
    d["county"] = d["home_county"].astype("category")

    # apply ordering
    d = apply_categorical_orders(d, CATEGORY_ORDERS)

    return d

def plot_teleworkable_trend(df):
    set_plot_style(base_size=20)

    trend = (
        df.loc[df["SOC_Teleworkable_Final"] == 1]
          .groupby(["SurveyYear", "arrangement"], as_index=False, observed=False)["w_trim"]
          .sum()
          .rename(columns={"w_trim": "weighted_n"})
    )

    trend["SurveyYear"] = pd.to_numeric(trend["SurveyYear"], errors="coerce")
    trend = trend.dropna(subset=["SurveyYear"]).copy()
    trend["SurveyYear"] = trend["SurveyYear"].astype(int)

    trend["pct"] = trend.groupby("SurveyYear")["weighted_n"].transform(
        lambda x: 100 * x / x.sum()
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    label_offsets = {
        "In-Person": {2023: 1.8, 2024: 1.8, 2025: -1.2},
        "Hybrid":    {2023: 1.8, 2024: 1.8, 2025:  1.8},
        "Remote":    {2023: 1.8, 2024: 1.8, 2025:  1.8},
    }

    for label in ARRANGEMENT_ORDER:
        sub = trend.loc[trend["arrangement"] == label].sort_values("SurveyYear")

        ax.plot(
            sub["SurveyYear"].to_numpy(),
            sub["pct"].to_numpy(),
            color=COLORS_ARRANGEMENT[label],
            linewidth=2,
            marker="o",
            markersize=8,
            label=label
        )

        for _, row in sub.iterrows():
            dy = label_offsets[label].get(int(row["SurveyYear"]), 1.5)
            ax.text(
                row["SurveyYear"],
                row["pct"] + dy,
                f"{row['pct']:.0f}%",
                color=COLORS_ARRANGEMENT[label],
                fontweight="bold",
                fontsize=16,
                ha="center"
            )

    ax.set_xlim(2022.8, 2025.2)
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xticklabels(["2023", "2024", "2025"])
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Work Arrangements among Teleworkable Workers, 2023–2025",
        loc="left",
        pad=10
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False
    )

    fig.text(
        0.01, 0.02,
        "Source: Maryland Commuter Survey, 2023–2025.\n"
        "Infrequent hybrid workers (2025) reclassified as in-person. "
        "Multiple worksite workers (2023–2024) reclassified as in-person.\n"
        "Weighted using post-stratification weights calibrated to 2020–2024 ACS 5-year estimates.",
        ha="left",
        va="bottom",
        fontsize=14
    )

    fig.subplots_adjust(top=0.90, bottom=0.22)
    return fig, ax

def plot_teleworkability_by_arrangement(df):
    set_plot_style(base_size=20)

    plot_data = (
        df.loc[df["SOC_Teleworkable_Final"].notna()]
          .groupby(["arrangement", "teleworkable_label"], as_index=False, observed=False)
          .agg(wt_n=("w_trim", "sum"), unwt_n=("w_trim", "size"))
    )
    plot_data["pct"] = plot_data.groupby("arrangement")["wt_n"].transform(lambda x: 100 * x / x.sum())

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(ARRANGEMENT_ORDER))

    for group in ["Non-Teleworkable", "Teleworkable"]:
        sub = (
            plot_data.loc[plot_data["teleworkable_label"] == group]
            .set_index("arrangement")
            .reindex(ARRANGEMENT_ORDER)
            .reset_index()
        )

        ax.bar(
            np.arange(len(ARRANGEMENT_ORDER)),
            sub["pct"].to_numpy(),
            bottom=bottom,
            width=0.65,
            color=COLORS_TELEWORKABLE[group],
            label=group
        )

        for i, row in sub.iterrows():
            if row["pct"] >= 3:
                ax.text(i, bottom[i] + row["pct"] * 0.55, f"{row['pct']:.0f}%",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=18)
                ax.text(i, bottom[i] + row["pct"] * 0.35, f"(n= {row['unwt_n']:,})",
                        ha="center", va="center", color="white", fontsize=14)

        bottom += sub["pct"].fillna(0).to_numpy()

    ax.set_xticks(np.arange(len(ARRANGEMENT_ORDER)))
    ax.set_xticklabels(ARRANGEMENT_ORDER)
    ax.set_ylim(0, 105)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks(np.arange(0, 80, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Work Arrangements by Dingel-Neiman Occupational Teleworkability", loc="left", pad=10)

    # fig.text(
    #     0.125, 0.91,
    #     "Employed Maryland Adults (Weighted Percentages)",
    #     ha="left", va="bottom", color="gray", fontsize=16
    # )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

    fig.text(
        0.01, 0.02,
        "Source: Maryland Commuter Survey, 2023–2025.\n"
        "Total sample includes 2,550 workers, of whom 159 have missing teleworkability data.\n"
        "Unweighted counts (n) shown in parentheses.\n"
        "Weighted using post-stratification weights calibrated to 2020–2024 ACS 5-year estimates.",
        ha="left", va="bottom", fontsize=14
    )

    fig.subplots_adjust(top=0.88, bottom=0.22)
    return fig, ax

DESC_BLOCKS = [
    ("Job Characteristics", [
        ("teleworkable_label", "Teleworkability (Dingel–Neiman)", ["Non-Teleworkable", "Teleworkable"]),
        ("major_soc_display", "Major occupation group", [
            "Management Occupations",
            "Office and Administrative Support Occupations",
            "Sales and Related Occupations",
            "Educational Instruction and Library Occupations",
            "Computer and Mathematical Occupations",
            "Food Preparation and Serving Related Occupations",
            "Business and Financial Operations Occupations",
            "Healthcare Practitioners and Technical Occupations",
            "Transportation and Material Moving Occupations",
            "Personal Care and Service Occupations",
            "Arts, Design, Entertainment, Sports, and Media Occupations",
            "Healthcare Support Occupations",
            "Construction and Extraction Occupations",
            "Protective Service Occupations",
            "Life, Physical, and Social Science Occupations",
            "Production Occupations",
            "Community and Social Service Occupations",
            "Architecture and Engineering Occupations",
            "Building and Grounds Cleaning and Maintenance Occupations",
            "Installation, Maintenance, and Repair Occupations",
            "Legal Occupations",
            "Farming, Fishing, and Forestry Occupations",
        ]),
        ("NAICS_industry_collapsed", "Industry", [
            'Professional/Finance/Information', 'Government',
       'Health Care', 'Consumer-Facing Services',
       'Construction/Manufacturing/Transport', 'Education',
       'Other Industries'
        ]),
        # ("desk_task", "Desk Work Tasks", [1, 0]),
        # ("interpersonal_task", "Interpersonal Work Tasks", [1, 0]),
        # ("physical_task", "Physical Work Tasks", [1, 0]),
        ("work_status_clean", "Employment status", ["Full-time", "Part-time", "Other/Flexible"]),
    ]),
    ("Preferences and Feasibility", [
        ("feasibility_display", "Self-reported remote feasibility", ["None", "Partial", "Full"]),
        ("preference_display", "Remote work preference", ["No remote", "Some remote", "Full remote"]),
    ]),
        ("Demographics", [
        ("age_display", "Age group", ["18–34", "35–54", "55+"]),
        ("gender_display", "Gender", ["Male", "Female"]),
        ("race_display", "Race/ethnicity", [
            "White non-Hispanic", "Black non-Hispanic", "Hispanic/Latinx", "Other/Multiracial"
        ]),
        ("income_display", "Household income", ["Under $50K", "$50K–$100K", "$100K–$150K", "Over $150K"]),
    ]),
    ("Household", [
        ("lifecycle_display", "Household type", ["Solitary", "Adults-Shared", "Family"]),
        ("vehicles_display", "Household vehicles", ["0 (car-free)", "1", "2", "3+"]),
    ]),
    ("Geography", [
        ("geo_display", "Home county type", ["Central county", "Outlying county"]),
        ("home_cbsa_display", "Home CBSA", ["Baltimore metro", "Washington metro", "Other/Non-metro"]),
        ("spatial_mismatch_display", "Home-work spatial mismatch", [
            "Same CBSA", "Cross-CBSA, same state", "Cross-state"
        ]),
        ("ruca_home_display", "Home RUCA type", [
            "Metro core", "Micro core", "Small town core", "Rural"
        ]),
        ("ruca_work_display", "Work RUCA type", [
            "Metro core", "Micro core", "Small town core", "Rural"
        ]),
    ]),
    ("", [
        ("year_display", "Survey year", ["2023", "2024", "2025"]),
    ]),
]

def build_descriptive_table(df):
    rows = []
    cols = ARRANGEMENT_ORDER + ["Total", "p"]

    rows.append({
        "Variable": "N (unweighted)",
        **{a: int((df["arrangement"] == a).sum()) for a in ARRANGEMENT_ORDER},
        "Total": len(df),
        "p": ""
    })
    rows.append({"Variable": "", **{c: "" for c in cols}})

    for block_name, variables in DESC_BLOCKS:
        if block_name:
            rows.append({"Variable": block_name, **{c: "" for c in cols}})

        for var, label, levels in variables:
            pval = weighted_chisq(df, var)
            rows.append({"Variable": label, **{c: "" for c in ARRANGEMENT_ORDER + ["Total"]}, "p": format_p(pval)})

            for level in levels:
                cell = {"Variable": f"    {level}", "p": ""}
                for a in ARRANGEMENT_ORDER:
                    sub = df.loc[df["arrangement"] == a]
                    pct = weighted_percent(sub[var], sub["w_trim"], level)
                    cell[a] = "—" if pd.isna(pct) else f"{pct:.1f}%"

                pct_total = weighted_percent(df[var], df["w_trim"], level)
                cell["Total"] = "—" if pd.isna(pct_total) else f"{pct_total:.1f}%"
                rows.append(cell)

            n_valid = df[var].notna().sum()
            if n_valid < len(df):
                rows.append({"Variable": f"    (N = {n_valid})", **{c: "" for c in cols}})

        rows.append({"Variable": "", **{c: "" for c in cols}})

    return pd.DataFrame(rows)

def fit_multinomial_model(df):
    formula = """
    C(arrangement, Treatment(reference='In-Person')) ~
    C(teleworkable, Treatment(reference='Not Teleworkable')) +
    C(industry, Treatment(reference='Consumer-Facing Services')) +
    C(income, Treatment(reference='$50-100k')) +
    C(age, Treatment(reference='35-54')) +
    C(education, Treatment(reference="Bachelor's")) +
    C(employment, Treatment(reference='Full-time')) +
    C(gender, Treatment(reference='Male')) +
    C(race, Treatment(reference='White')) +
    C(household, Treatment(reference='Adults-Shared')) +
    C(vehicles, Treatment(reference='0-1')) +
    C(county_type, Treatment(reference='Central')) +
    C(home_cbsa, Treatment(reference='Baltimore')) +
    C(year, Treatment(reference='2023'))
    """

    model_vars = [
        "arrangement", "teleworkable", "industry", "income", "age", "education",
        "employment", "gender", "race", "household", "vehicles",
        "county_type", "home_cbsa", "year", "county"
    ]

    d = df.dropna(subset=model_vars).copy()

    y, X = patsy.dmatrices(formula, d, return_type="dataframe")
    d = d.loc[y.index].copy()

    y_cat = d["arrangement"].astype("category")
    outcome_levels = list(y_cat.cat.categories)
    y_codes = y_cat.cat.codes

    model = sm.MNLogit(y_codes, X)
    result = model.fit(method="newton", maxiter=200, disp=False)

    return result, X, y_cat, d, outcome_levels

def clustered_covariance(result, groups, small_sample=True):
    score_obs = result.model.score_obs(result.params.to_numpy().ravel())
    K = score_obs.shape[1]

    score_df = pd.DataFrame(score_obs)
    score_df["group"] = pd.Series(groups).reset_index(drop=True).values
    U = score_df.groupby("group", sort=False).sum()
    meat = U.to_numpy().T @ U.to_numpy()

    bread = result.cov_params().to_numpy()
    V = bread @ meat @ bread

    if small_sample:
        G = pd.Series(groups).nunique()
        N = len(groups)
        correction = (G / (G - 1)) * ((N - 1) / (N - K))
        V *= correction

    return V

def tidy_multinomial_results(result, V, df_t, outcome_levels, reference="In-Person"):
    crit = t.ppf(0.975, df=df_t)
    beta = result.params.copy()

    rows = []
    se_all = np.sqrt(np.diag(V))
    param_index = []

    for outcome in beta.columns:
        for term in beta.index:
            param_index.append((outcome, term))

    for (outcome, term), se in zip(param_index, se_all):
        est = beta.loc[term, outcome]
        tval = est / se
        pval = 2 * t.sf(abs(tval), df=df_t)
        rows.append({
            "outcome": outcome,
            "term": term,
            "estimate": est,
            "se": se,
            "t": tval,
            "p": pval,
            "OR": np.exp(est),
            "OR_lo": np.exp(est - crit * se),
            "OR_hi": np.exp(est + crit * se),
        })

    out = pd.DataFrame(rows)

    non_base_levels = [x for x in outcome_levels if x != reference]
    outcome_map = {col: lab for col, lab in zip(beta.columns, non_base_levels)}
    out["outcome"] = out["outcome"].map(outcome_map)

    return out

def model_fit_summary(result, df, outcome_levels, outcome="arrangement"):
    pred_probs = np.asarray(result.predict())
    pred_codes = pred_probs.argmax(axis=1)

    pred_labels = pd.Series(
        pd.Categorical.from_codes(pred_codes, categories=outcome_levels)
    ).astype(str)

    observed = df[outcome].reset_index(drop=True).astype(str)
    accuracy = (pred_labels.to_numpy() == observed.to_numpy()).mean()

    cm = pd.crosstab(observed, pred_labels, dropna=False)
    recall = np.diag(cm.div(cm.sum(axis=1), axis=0).fillna(0).to_numpy())
    bal_acc = np.nanmean(recall)

    return pd.DataFrame([{
        "N": len(df),
        "Counties": df["county"].nunique(),
        "logLik": round(result.llf, 1),
        "AIC": round(result.aic, 1),
        "BIC": round(result.bic, 1),
        "Accuracy": round(accuracy, 3),
        "BalancedAccuracy": round(bal_acc, 3),
    }])

def simplify_term(term):
    lookup = {
        "Intercept": "(Intercept)",
        "C(teleworkable, Treatment(reference='Not Teleworkable'))[T.Teleworkable]": "teleworkableTeleworkable",
        "C(industry, Treatment(reference='Retail/Food/Hospitality'))[T.Professional/Finance/Information]": "industryProfessional/Finance/Information",
        "C(industry, Treatment(reference='Retail/Food/Hospitality'))[T.Health Care]": "industryHealth Care",
        "C(industry, Treatment(reference='Retail/Food/Hospitality'))[T.Education]": "industryEducation",
        "C(industry, Treatment(reference='Retail/Food/Hospitality'))[T.Government]": "industryGovernment",
        "C(industry, Treatment(reference='Retail/Food/Hospitality'))[T.Construction/Manufacturing/Transport]": "industryConstruction/Manufacturing/Transport",
        "C(income, Treatment(reference='$50-100k'))[T.<$50k]": "income<$50k",
        "C(income, Treatment(reference='$50-100k'))[T.$100-150k]": "income$100-150k",
        "C(income, Treatment(reference='$50-100k'))[T.>$150k]": "income>$150k",
        'C(education, Treatment(reference="Bachelor\'s"))[T.Pre-bachelor]': "educationPre-bachelor",
        'C(education, Treatment(reference="Bachelor\'s"))[T.Graduate]': "educationGraduate",
        "C(employment, Treatment(reference='Full-time'))[T.Not full-time]": "employmentNot full-time",
        "C(age, Treatment(reference='35-54'))[T.18-34]": "age18-34",
        "C(age, Treatment(reference='35-54'))[T.55+]": "age55+",
        "C(gender_clean, Treatment(reference='Male'))[T.Female]": "genderFemale",
        "C(race, Treatment(reference='White'))[T.Black]": "raceBlack",
        "C(race, Treatment(reference='White'))[T.Hispanic/Latinx]": "raceHispanic/Latinx",
        "C(race, Treatment(reference='White'))[T.Other]": "raceOther",
        "C(household, Treatment(reference='Adults-Shared'))[T.Solitary]": "householdSolitary",
        "C(household, Treatment(reference='Adults-Shared'))[T.Family]": "householdFamily",
        "C(vehicles, Treatment(reference='0-1'))[T.2+]": "vehicles2+",
        "C(county_type, Treatment(reference='Central'))[T.Outlying]": "county_typeOutlying",
        "C(home_cbsa, Treatment(reference='Baltimore'))[T.Washington]": "home_cbsaWashington",
        "C(home_cbsa, Treatment(reference='Baltimore'))[T.Other]": "home_cbsaOther",
        "C(year, Treatment(reference='2023'))[T.2024]": "year2024",
        "C(year, Treatment(reference='2023'))[T.2025]": "year2025",
    }
    return lookup.get(term, term)

def prepare_or_plot_data(tidy_df, p_cutoff=0.10):
    out = tidy_df.copy()
    out = out.loc[out["term"] != "Intercept"].copy()
    out["term"] = out["term"].map(simplify_term)
    out = out.loc[out["term"].isin(MODEL_LABELS)].copy()
    out["term_label"] = out["term"].map(MODEL_LABELS)
    out["label"] = np.where(out["p"] < p_cutoff, out["OR"].map(lambda x: f"{x:.2f}") + out["p"].map(p_stars), "")
    out["fontweight"] = np.where(out["p"] < 0.05, "bold", "normal")
    out["x_label"] = out.groupby("term")["OR_hi"].transform("max") * 1.08
    out["term_label"] = pd.Categorical(out["term_label"], categories=list(reversed(MODEL_TERM_ORDER)), ordered=True)
    out = out.loc[out["p"] < p_cutoff].copy()
    return out


def plot_or_forest(plot_df):
    set_plot_style(base_size=15)

    fig, ax = plt.subplots(figsize=(12, 10))
    y_levels = list(plot_df["term_label"].cat.categories)
    y_map = {v: i for i, v in enumerate(y_levels)}
    offsets = {"Hybrid": -0.15, "Remote": 0.15}
    markers = {"Hybrid": "s", "Remote": "o"}

    ax.axvline(1, linestyle="--", color="gray", linewidth=1)

    for outcome in ["Hybrid", "Remote"]:
        sub = plot_df.loc[plot_df["outcome"] == outcome].copy()
        y = sub["term_label"].map(y_map).astype(float) + offsets[outcome]

        ax.hlines(y, sub["OR_lo"], sub["OR_hi"], color=COLORS_MODEL[outcome], linewidth=2, alpha=0.9)
        ax.scatter(sub["OR"], y, color=COLORS_MODEL[outcome], marker=markers[outcome], s=70, label=outcome)

        for _, row in sub.iterrows():
            if row["label"]:
                ax.text(row["x_label"], y_map[row["term_label"]] + offsets[outcome], row["label"],
                        ha="left", va="center", fontsize=14, fontweight=row["fontweight"])

    ax.set_xscale("log")
    ax.set_xlim(0.15, 15)
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels(y_levels)
 #   ax.set_xlabel("Odds Ratio (Log Scale)")
    ax.set_ylabel("")
   # ax.set_title("Model 1A: Correlates of Work Arrangement", loc="left")
    ax.text(
        0, 1.02,
        "Odds Ratios with County-Clustered 95% CIs (Log Scale).\nReference: In-Person Worker. Full Sample.",
        transform=ax.transAxes,
        ha="left", va="bottom", color="gray"
    )
    ax.legend(title="Outcome vs In-Person", loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    fig.text(
        0.01, 0.01,
        "Reference: In-Person. Not Teleworkable; Retail/Food/Hospitality; $50-100k; Bachelor's; Full-time;\n"
        "35-54; Male; White; Adults-Shared; 0-1 Vehicles; Central County; Baltimore Metro; 2023.\n"
        "Note: † p<0.10; * p<0.05; ** p<0.01; *** p<0.001. Labels shown for p<0.10.",
        ha="left", va="bottom", fontsize=10
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, ax

def make_or_table(tidy_df):
    out = tidy_df.copy()
    out = out.loc[out["term"] != "Intercept"].copy()
    out["term"] = out["term"].map(simplify_term)
    out = out.loc[out["term"].isin(MODEL_LABELS)].copy()
    out["term_label"] = out["term"].map(MODEL_LABELS)
    out["OR"] = out["OR"].map(lambda x: f"{x:.2f}") + out["p"].map(p_stars)
    out["SE"] = out["se"].map(lambda x: f"{x:.2f}")
    out["CI"] = out.apply(lambda r: f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]", axis=1)
    table = (
        out[["term_label", "outcome", "OR", "SE", "CI"]]
        .pivot(index="term_label", columns="outcome", values=["OR", "SE", "CI"])
    )
    table.columns = [f"{outcome}_{stat}" for stat, outcome in table.columns]
    return table.reset_index()