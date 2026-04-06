#!/usr/bin/env python
# coding: utf-8

# In[692]:


import numpy as np
import pandas as pd
import string
import os
import re
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from adjustText import adjust_text
import patsy
import statsmodels.api as sm
from scipy.stats import chi2, t
import geopandas as gpd
import folium
from folium.plugins import HeatMapWithTime

import geo_vars
from modules.config import load_configurations
import Survey_Weighting_V2 as srw
import Google_API_Metrics as g_api_commute
import ses_demographic_data_build as ses
import vars_harmonization as varh
import industry_module as industry
import work_activities
from statsmodels.nonparametric.smoothers_lowess import lowess


# In[704]:


import Google_API_Metrics as g_api_commute


# In[7]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[19]:


configs = load_configurations()
mcs_2023 = os.path.join(configs.processed_data_drive_path, 'mcs_2023.csv')
mcs_2024 = os.path.join(configs.processed_data_drive_path, 'mcs_2024.csv')


# In[21]:


mcs_2023 = pd.read_csv(mcs_2023)
mcs_2024 = pd.read_csv(mcs_2024)
mcs_2025 = pd.read_csv('data/mcs_2025_main_table.csv')


# In[196]:


SOC_teleworkability = pd.read_csv('data/processed/processed/final_SOC_teleworkable_dataset_Jan7.csv')


# In[194]:


major_SOC = pd.read_csv('Major SOC Codes.csv')


# In[150]:


cbsa_data = pd.read_csv("data/processed/processed/cbsa_data.csv")


# In[184]:


RUCA_data = pd.read_csv('RUCA Codes 2020.csv', encoding="cp1252")


# In[23]:


cols = ['record_id', 'work_remote']
stated_telework = pd.concat([mcs_2023[cols], mcs_2024[cols], mcs_2025[cols]])


# In[25]:


stated_telework.head().T


# In[27]:


stated_telework['work_remote_clean'] = pd.Categorical(
    stated_telework["work_remote"].map(varh.work_remote_3cat_map),
    categories=["fully_in_person", "hybrid", "fully_remote"],
    ordered=True
)


# In[29]:


stated_telework['work_remote_clean'].value_counts(dropna=False)


# In[37]:


stated_telework.shape


# In[33]:


stated_telework = stated_telework.query("work_remote_clean.notna()")


# In[35]:


stated_telework.shape


# In[39]:


cols = ['record_id','hh_income', 'education', 'age', 'gender', 'race_ethnicity', 'race_ethnicity_text', 'work_status', 'hh_size','hh_children','hh_vehicles']
ses_demographic_panel = pd.concat([mcs_2023[cols], mcs_2024[cols], mcs_2025[cols]])
ses_demographic_panel = ses.DemographicFactors.build_all(ses_demographic_panel)


# In[41]:


ses_demographic_panel.head().T


# In[47]:


ses_demographic_panel.columns.tolist()


# In[264]:


cols = ['record_id', 'household_composition','income_clean', 'education_clean', 'age_clean', 'gender_clean','race_clean', 'work_status_clean','hh_vehicles_clean']
model_df = stated_telework.merge(ses_demographic_panel[cols], how = 'inner', on = 'record_id')


# In[266]:


model_df.head().T


# In[268]:


mcs_2025['SurveyYear'] = 2025


# In[270]:


cols = ['record_id', 'remote_potential', 'remote_preference', 'SurveyYear']
remote_vars = pd.concat([mcs_2023[cols],mcs_2024[cols],mcs_2025[cols]])


# In[210]:


remote_vars.head().T


# In[272]:


remote_vars.remote_preference.value_counts(dropna=False)


# In[274]:


model_df = model_df.merge(remote_vars, how = 'inner', on = 'record_id')


# In[276]:


model_df.loc[(model_df.work_remote.eq('remote')) & (model_df.remote_potential.isna()), 'remote_potential'] = 'full_remote_potential'


# In[278]:


order = ["no_remote_potential", "partial_remote_potential", "full_remote_potential"]
model_df["remote_potential"] = pd.Categorical(model_df["remote_potential"], categories=order, ordered=True)


# In[280]:


order = ["no_remote_prefer", "partial_remote_prefer", "full_remote_prefer"]
model_df["remote_preference"] = pd.Categorical(model_df["remote_preference"], categories=order, ordered=True)


# In[282]:


pd.crosstab(model_df.remote_preference, model_df.work_remote_clean, normalize='columns').round(2)*100


# In[284]:


cols = ['record_id', 'work_industry', 'work_industry_text']
work_industries = pd.concat([mcs_2023[cols], mcs_2024[cols], mcs_2025[cols]], ignore_index=True)


# In[286]:


df = industry.WorkIndustryFactors.apply(
    work_industries,
    work_industry_col="work_industry",
    text_col="work_industry_text"
)


# In[288]:


df.head().T


# In[290]:


summary_df = industry.WorkIndustryFactors.summary_table(df)


# In[292]:


summary_df


# In[294]:


model_df = model_df.merge(df[[
     'record_id',
     'NAICS_industry',
     'NAICS_industry_collapsed',
]], 
  how = 'left', 
  on='record_id')


# In[296]:


model_df.shape


# In[298]:


SOC_teleworkability.head().T


# In[300]:


cols = ['record_id', 'SOC_Teleworkable_Final', "SOC_Code_Final", 'SOC_Title_Final', 'Match_Type_Final']
model_df = model_df.merge(SOC_teleworkability[cols], on = 'record_id', how ='left')


# In[302]:


model_df.SOC_Teleworkable_Final.value_counts(dropna=False)


# In[244]:


# model_df = model_df.query("SOC_Teleworkable_Final.notna()")


# In[304]:


major_SOC["Major_SOC_Code"] = pd.to_numeric(major_SOC["Major_SOC_Code"], errors="coerce").astype("Int64")


# In[306]:


teleworkable_flag = pd.read_csv('D_N_SOC.csv')
teleworkable_flag = teleworkable_flag.add_suffix("_DN_SOC")


# In[308]:


teleworkable_flag.columns


# In[310]:


model_df = model_df.merge(teleworkable_flag, how ='left', left_on = 'SOC_Title_Final', right_on = 'title_DN_SOC')
model_df['SOC_Code_Final'] = model_df['SOC_Code_Final'].combine_first(model_df['onetsoccode_DN_SOC'])
model_df['SOC_Teleworkable_Final'] = model_df['SOC_Teleworkable_Final'].combine_first(model_df['teleworkable_DN_SOC'])


# In[312]:


model_df["Major_SOC_Code"] = model_df["SOC_Code_Final"].str.split("-", n=1).str[0].str.strip()
model_df["Major_SOC_Code"] = pd.to_numeric(model_df["Major_SOC_Code"], errors="coerce").astype("Int64")
model_df = model_df.merge(major_SOC, on ='Major_SOC_Code', how = 'left')


# In[326]:


model_df.columns.tolist()


# In[320]:


cols = [
 'SOC_Teleworkable_Final',
 'SOC_Code_Final',
 'SOC_Title_Final',
 'Match_Type_Final',
 # 'onetsoccode_DN_SOC',
 # 'title_DN_SOC',
 # 'teleworkable_DN_SOC',
 'Major_SOC_Code',
 'Major_SOC_Occupations']


# In[322]:


[model_df[col].value_counts(dropna=False) for col in cols]


# In[324]:


model_df = model_df.drop([ 
 'onetsoccode_DN_SOC',
 'title_DN_SOC',
 'teleworkable_DN_SOC'], axis=1)


# In[182]:


cols = ['record_id','work_location_lat','work_location_lon','home_location_lat', 'home_location_lon']
locations_df = pd.concat([mcs_2023[cols], mcs_2024[cols], mcs_2025[cols]])


# In[186]:


locations_df = geo_vars.home_geo_info(locations_df, RUCA_data, cbsa_data)


# In[342]:


locations_df.head().T


# In[192]:


locations_df.loc[locations_df.home_cbsa_code.isna(), 'home_cbsa_title'] = 'Outside_CBSA_or_Unclassified'
locations_df.loc[locations_df.home_cbsa_code.isna(), 'home_central_outlying_county'] = 'Outside_CBSA_or_Unclassified'


# In[334]:


locations_df = geo_vars.work_geo_info(locations_df, RUCA_data, cbsa_data)


# In[340]:


locations_df.work_state.value_counts(dropna=False)


# In[344]:


locations_df.loc[locations_df.work_cbsa_code.isna(), 'work_cbsa_title'] = 'Outside_CBSA_or_Unclassified'
locations_df.loc[locations_df.work_cbsa_code.isna(), 'work_central_outlying_county'] = 'Outside_CBSA_or_Unclassified'


# In[338]:


locations_df = geo_vars.build_home_work_spatial_mismatch(locations_df)


# In[346]:


locations_df.columns


# In[352]:


cols = ['hw_same_cbsa', 'hw_same_state',
       'hw_same_county', 'hw_boundary_type', 'hw_spatial_mismatch',
       'hw_major_boundary']


# In[360]:


[locations_df[col].value_counts() for col in cols]


# In[368]:


model_df = model_df.merge(locations_df, how='left', on='record_id')


# In[370]:


model_df.work_remote_clean.value_counts()


# In[378]:


model_df[['home_location_lon', 'home_location_lat', 'work_remote_clean']].isna().sum()


# In[384]:


model_df.columns.tolist()


# In[406]:


tracts_md = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_24_tract.zip")
tracts_dc = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_11_tract.zip")

tracts_mddc = pd.concat([tracts_md, tracts_dc], ignore_index=True)
tracts_mddc = gpd.GeoDataFrame(tracts_mddc, geometry="geometry", crs=tracts_md.crs).to_crs(epsg=4326)

tracts_display = tracts_mddc[["GEOID", "STATEFP", "COUNTYFP", "NAMELSAD", "geometry"]].copy()

remote_df = model_df.loc[
    model_df["work_remote_clean"].eq("fully_remote", "hybrid")
    & model_df["home_location_lat"].notna()
    & model_df["home_location_lon"].notna()
    & model_df["SurveyYear"].notna(),
    ["SurveyYear", "home_location_lat", "home_location_lon"]
].copy()

years = sorted(remote_df["SurveyYear"].astype(int).unique())

center_lat = remote_df["home_location_lat"].mean()
center_lon = remote_df["home_location_lon"].mean()

heat_data = []
time_index = []

for year in years:
    year_df = remote_df.loc[remote_df["SurveyYear"].astype(int) == year].copy()
    points = year_df[["home_location_lat", "home_location_lon"]].values.tolist()
    heat_data.append(points)
    time_index.append(str(year))

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="CartoDB positron"
)

folium.GeoJson(
    tracts_display.to_json(),
    name="MD + DC Census Tracts",
    style_function=lambda x: {
        "fillColor": "#3186cc" if x["properties"]["STATEFP"] == "24" else "#cc4331",
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.08,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["GEOID", "STATEFP", "COUNTYFP", "NAMELSAD"],
        aliases=["GEOID", "STATEFP", "COUNTYFP", "Tract"]
    )
).add_to(m)

HeatMapWithTime(
    data=heat_data,
    index=time_index,
    name="Remote worker heatmap",
    auto_play=False,
    max_opacity=0.8,
    radius=18,
    use_local_extrema=False
).add_to(m)

folium.LayerControl().add_to(m)
m


# In[412]:


df_weighted, diagnostics, county_emp_tbl = srw.run_full_weighting_pipeline(
    model_df,
    acs_year=2024,
    api_key="",
    rake_max_iter=150,
    rake_tol=1e-3,
)


# In[422]:


diagnostics = srw.weight_diagnostics(df_weighted)
diagnostics[["SurveyYear", "N", "sum_w", "mean_w"]]


# In[424]:


county_edu_tbl = srw.fetch_education_by_county(2024, api_key="9cb57f8ef6deecfe7c23bc45c30300c9604757e1")

county_to_geo = (
    df_weighted.dropna(subset=["county_clean", "geo_bin"])
    .groupby("county_clean")["geo_bin"]
    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Outlying")
)
year_levels = sorted(df_weighted["SurveyYear"].dropna().astype(str).unique())
target_counties = sorted(df_weighted["county_clean"].dropna().unique())

demo_targets = srw.build_demo_targets(
    county_employed_tbl=county_emp_tbl,
    county_to_geo=county_to_geo,
    target_counties=target_counties,
    year_levels=year_levels,
)
demo_targets["pop_total_y"] = demo_targets.groupby("SurveyYear")["n_pop"].transform("sum")
demo_targets["demo_share"] = demo_targets["n_pop"] / demo_targets["pop_total_y"]

edu_targets = srw.build_edu_targets(
    county_edu_tbl=county_edu_tbl,
    county_to_geo=county_to_geo,
    target_counties=target_counties,
    year_levels=year_levels,
)

mc_demo = srw.margin_check(
    df_weighted,
    margin_keys=["geo_bin", "race_bin", "age_bin"],
    targets=demo_targets,
    target_col="demo_share",
    w_col="w_trim",
    year_col="SurveyYear",
)

mc_edu = srw.margin_check(
    df_weighted,
    margin_keys=["edu_bin"],
    targets=edu_targets,
    target_col="edu_share",
    w_col="w_trim",
    year_col="SurveyYear",
)

unsupported_demo_by_year, unsupported_edu_by_year = srw.collect_unsupported_cells_by_year(
    df_weighted,
    county_emp_tbl=county_emp_tbl,
    county_edu_tbl=county_edu_tbl,
    year_col="SurveyYear",
)

weighting_summary_tbl = srw.summarize_weighting_results(
    df_w=df_weighted,
    demo_margin_check=mc_demo,
    edu_margin_check=mc_edu,
    unsupported_demo_by_year=unsupported_demo_by_year,
    unsupported_edu_by_year=unsupported_edu_by_year,
    year_col="SurveyYear",
    w_col="w_trim",
)
weighting_summary_tbl.T


# In[754]:


import multinomial_regression as mlr


# In[758]:


df_weighted['work_remote_clean_'] = df_weighted['work_remote_clean']
df_weighted.loc[df_weighted['work_remote'] == 'infrequent_hybrid', 'work_remote_clean_'] = 'fully_in_person'


# In[760]:


analysis_df = mlr.prepare_analysis_data(df_weighted)
analysis_df.columns.tolist()


# In[762]:


fig1, ax1 = mlr.plot_teleworkable_trend(analysis_df)
plt.show()


# In[764]:


fig2, ax2 = mlr.plot_teleworkability_by_arrangement(analysis_df)
plt.show()


# In[766]:


desc_table = mlr.build_descriptive_table(analysis_df)
print(desc_table.to_string(index=False))


# In[768]:


m1_result, X, y, m1_data, outcome_levels = mlr.fit_multinomial_model(analysis_df)
m1_fit_stats = mlr.model_fit_summary(m1_result, m1_data, outcome_levels)
print(m1_fit_stats)


# In[770]:


V_cluster = mlr.clustered_covariance(m1_result, m1_data["county"], small_sample=True)
df_t = m1_data["county"].nunique() - 1
m1_tidy = mlr.tidy_multinomial_results(m1_result, V_cluster, df_t, outcome_levels)


# In[780]:


print(m1_fit_stats)
plot_df = mlr.prepare_or_plot_data(m1_tidy, p_cutoff=0.10)


# In[782]:


plot_df = mlr.prepare_or_plot_data(m1_tidy, p_cutoff=0.10)
fig3, ax3 = mlr.plot_or_forest(plot_df)


# In[776]:


or_table = mlr.make_or_table(m1_tidy)
print(or_table.to_string(index=False))


# In[ ]:


api_key = ''
api_cfg = g_api_commute.CommuteAPIScenarioConfig()
gmaps_client = googlemaps.Client(
    key=api_key,
    queries_per_second=3,
    retry_over_query_limit=True,
    timeout=10,
)
api_results = g_api_commute.build_api_commute_measures(
    df=model_df,
    gmaps_client=gmaps_client,
    config=api_cfg,
    origin_lat_col="home_location_lat",
    origin_lon_col="home_location_lon",
    dest_lat_col="work_location_lat",
    dest_lon_col="work_location_lon",
    primary_mode_col="primary_mode_collapsed",
    primary_time_window_col="primary_time_window",
)
df_api = pd.concat([model_df, api_results], axis=1)


# In[680]:


tt_distance_data = pd.read_csv('model_data_V1_0504.csv')


# In[690]:


tt_distance_data.api_distance_miles_clean.describe()


# In[698]:


tt_distance_data = tt_distance_data.assign(
    commute_days=tt_distance_data["commute_days_clean"],
    work_days=tt_distance_data["work_days_clean_imputed"],
)

tt_distance_data["work_days"] = np.where(
    tt_distance_data["commute_days"].notna()
    & tt_distance_data["work_days"].notna()
    & (tt_distance_data["commute_days"] > tt_distance_data["work_days"]),
    tt_distance_data["commute_days"],
    tt_distance_data["work_days"],
)

tt_distance_data["commute_day_share"] = np.where(
    (tt_distance_data["work_days"] > 0) & tt_distance_data["commute_days"].notna(),
    np.minimum(tt_distance_data["commute_days"] / tt_distance_data["work_days"], 1),
    np.nan,
)


# In[702]:


plot_df = tt_distance_data[["api_distance_miles_clean", "commute_day_share"]].copy()
plot_df = plot_df.dropna()

x = plot_df["api_distance_miles_clean"].to_numpy()
y = plot_df["commute_day_share"].to_numpy()

smoothed = lowess(y, x, frac=0.3, return_sorted=True)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.25)
plt.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2)

plt.xlabel("API commute distance (miles)")
plt.ylabel("Commute day share")
plt.title("Commute day share vs API commute distance")
plt.ylim(0, 1.05)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()

