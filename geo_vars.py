import pandas as pd
import numpy as np
import reverse_geocoder as rg
import geopandas as gpd

def home_geo_info(df, RUCA, cbsa):
    mask = df['home_location_lat'].notna() & df['home_location_lon'].notna()
    coords_to_search = list(zip(df.loc[mask, 'home_location_lat'], 
                            df.loc[mask, 'home_location_lon']))
    if len(coords_to_search) > 0:
        results = rg.search(coords_to_search)
        df.loc[mask, 'home_state'] = [x['admin1'] for x in results]
        df.loc[mask, 'home_county'] = [x['admin2'] for x in results]
        df.loc[mask, 'home_city'] = [x['name'] for x in results]

    tracts_url = (
    "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
    "tl_2023_24_tract.zip")

    df = df[df.home_state=='Maryland']

    tracts = gpd.read_file(tracts_url)
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["home_location_lon"], 
                                    df["home_location_lat"]),
        crs="EPSG:4326"
    )
    points_gdf = points_gdf.to_crs(tracts.crs)
    points_with_tracts = gpd.sjoin(
        points_gdf,
        tracts[["GEOID", "NAMELSAD", "COUNTYFP", "TRACTCE", "geometry"]],
        how="left",
        predicate="within"
    )
    points_with_tracts = points_with_tracts.drop(columns=["index_right"])
    RUCA["TractFIPS23"] = (
    pd.to_numeric(RUCA["TractFIPS23"], errors="coerce")
    .astype("Int64")
    .astype(str)
    .str.zfill(11)
    )
    points_with_tracts["GEOID"] = points_with_tracts["GEOID"].astype(str)
    RUCA = RUCA.rename(
    columns={c: f"ruca_home_{c}" for c in RUCA.columns if c != "TractFIPS23"}
    )
    df = points_with_tracts.merge(
                                RUCA, 
                                how='left', 
                                left_on='GEOID', 
                                right_on='TractFIPS23',
                                suffixes=("", "ruca_home"))
    
    geo_cols = ['home_state', 'home_county', 'home_city']
    df[geo_cols] = df[geo_cols].replace('nan', np.nan)
    df[geo_cols] = df[geo_cols].replace('', 'District of Columbia')
    df.loc[df.home_state == 'Washington, D.C.', 'home_state'] = 'District of Columbia'
    df['home_county'] = df['home_county'].str.replace(r'^City of\s+(.*)', r'\1 city', regex=True)
    
    df = df.merge(cbsa[[
                        'CBSA Code',
                        'CBSA Title',
                        'County/County Equivalent',
                        'State Name',
                        'Central/Outlying County']],                           
                how = 'left', 
                left_on = ['home_county', 'home_state'], 
                right_on = ['County/County Equivalent','State Name'] )
    
    cols_to_rename = [
        'CBSA Code',
        'CBSA Title',
        'County/County Equivalent',
        'State Name',
        'Central/Outlying County'
        ]
    rename_map = {
            col: f"home_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in cols_to_rename
        }
    df = df.rename(columns=rename_map)
    return df

def work_geo_info(df, RUCA, cbsa):
    mask = df['work_location_lat'].notna() & df['work_location_lon'].notna()
    coords_to_search = list(zip(df.loc[mask, 'work_location_lat'], 
                            df.loc[mask, 'work_location_lon']))
    if len(coords_to_search) > 0:
        results = rg.search(coords_to_search)
        df.loc[mask, 'work_state'] = [x['admin1'] for x in results]
        df.loc[mask, 'work_county'] = [x['admin2'] for x in results]
        df.loc[mask, 'work_city'] = [x['name'] for x in results]

    state_name_to_fips = {
        "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
        "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
        "District of Columbia": "11", "Washington, D.C.": "11",
        "Florida": "12", "Georgia": "13", "Hawaii": "15", "Idaho": "16",
        "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
        "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24",
        "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28",
        "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
        "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
        "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
        "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
        "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49",
        "Vermont": "50", "Virginia": "51", "Washington": "53", "West Virginia": "54",
        "Wisconsin": "55", "Wyoming": "56"
    }
    work_states = (
        df["work_state"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"nan": pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    work_state_fips = [state_name_to_fips[s] for s in work_states if s in state_name_to_fips]

    tract_list = []

    for fips in sorted(set(work_state_fips)):
        url = f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{fips}_tract.zip"
        gdf = gpd.read_file(url)
        tract_list.append(gdf)
    
    tracts_workstates = pd.concat(tract_list, ignore_index=True)
    tracts_workstates = gpd.GeoDataFrame(
        tracts_workstates,
        geometry="geometry",
        crs=tract_list[0].crs
    )
    
    tracts_workstates["GEOID"] = tracts_workstates["GEOID"].astype(str)
    tracts_workstates["STATEFP"] = tracts_workstates["STATEFP"].astype(str)

    work_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["work_location_lon"], 
                                df["work_location_lat"]),
    crs="EPSG:4326"
    )

    if work_points.crs != tracts_workstates.crs:
        work_points = work_points.to_crs(tracts_workstates.crs)
    
    work_points_tracts = gpd.sjoin(
        work_points,
        tracts_workstates[["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAMELSAD", "geometry"]],
        how="left",
        predicate="within"
    ).drop(columns="index_right", errors="ignore")

    RUCA_work = RUCA.rename(
        columns={c: f"ruca_work_{c}" for c in RUCA.columns if c != "TractFIPS23"}
    )

    df = work_points_tracts.merge(RUCA_work, 
                                how='left', 
                                left_on='GEOID_right', 
                                right_on='TractFIPS23')

    geo_cols = ['work_state', 'work_county', 'work_city']
    df[geo_cols] = df[geo_cols].replace('nan', np.nan)
    df[geo_cols] = df[geo_cols].replace('', 'District of Columbia')
    df.loc[df.home_state == 'Washington, D.C.', 'work_state'] = 'District of Columbia'
    df['work_county'] = df['work_county'].str.replace(r'^City of\s+(.*)', r'\1 city', regex=True)
    
    df = df.merge(cbsa[[
                        'CBSA Code',
                        'CBSA Title',
                        'County/County Equivalent',
                        'State Name',
                        'Central/Outlying County']],                           
                how = 'left', 
                left_on = ['work_county', 'work_state'], 
                right_on = ['County/County Equivalent','State Name'] )
    
    cols_to_rename = [
        'CBSA Code',
        'CBSA Title',
        'County/County Equivalent',
        'State Name',
        'Central/Outlying County'
        ]
    rename_map = {
            col: f"work_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in cols_to_rename
        }
    df = df.rename(columns=rename_map)
    return df

def _norm_str(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip()
    if s == "":
        return pd.NA
    return s

def build_home_work_spatial_mismatch(
    df: pd.DataFrame,
    home_cbsa_col: str = "home_cbsa_code",
    work_cbsa_col: str = "work_cbsa_code",
    home_state_col: str = "home_state",
    work_state_col: str = "work_state",
    home_county_col: str | None = "home_county",
    work_county_col: str | None = "work_county",
) -> pd.DataFrame:
    """
    Construct home-work spatial mismatch indicators.
    """

    out = df.copy()

    home_cbsa = out[home_cbsa_col].map(_norm_str)
    work_cbsa = out[work_cbsa_col].map(_norm_str)

    home_state = out[home_state_col].map(_norm_str).str.upper()
    work_state = out[work_state_col].map(_norm_str).str.upper()

    if home_county_col is not None and work_county_col is not None:
        home_county = out[home_county_col].map(_norm_str).str.upper()
        work_county = out[work_county_col].map(_norm_str).str.upper()
    else:
        home_county = pd.Series(pd.NA, index=out.index, dtype="object")
        work_county = pd.Series(pd.NA, index=out.index, dtype="object")

    out["hw_same_cbsa"] = np.where(
        home_cbsa.isna() | work_cbsa.isna(),
        pd.NA,
        (home_cbsa == work_cbsa).astype("int8")
    )

    out["hw_same_state"] = np.where(
        home_state.isna() | work_state.isna(),
        pd.NA,
        (home_state == work_state).astype("int8")
    )

    out["hw_same_county"] = np.where(
        home_county.isna() | work_county.isna(),
        pd.NA,
        (home_county == work_county).astype("int8")
    )

    same_cbsa = pd.Series(out["hw_same_cbsa"], index=out.index)
    same_state = pd.Series(out["hw_same_state"], index=out.index)

    boundary_type = pd.Series(pd.NA, index=out.index, dtype="object")

    missing_mask = same_cbsa.isna() | same_state.isna()

    boundary_type = boundary_type.mask(
        ~missing_mask & (same_cbsa == 1) & (same_state == 1),
        "same_cbsa_same_state"
    )
    boundary_type = boundary_type.mask(
        ~missing_mask & (same_cbsa == 1) & (same_state == 0),
        "same_cbsa_cross_state"
    )
    boundary_type = boundary_type.mask(
        ~missing_mask & (same_cbsa == 0) & (same_state == 1),
        "cross_cbsa_same_state"
    )
    boundary_type = boundary_type.mask(
        ~missing_mask & (same_cbsa == 0) & (same_state == 0),
        "cross_cbsa_cross_state"
    )

    out["hw_boundary_type"] = pd.Categorical(
        boundary_type,
        categories=[
            "same_cbsa_same_state",
            "same_cbsa_cross_state",
            "cross_cbsa_same_state",
            "cross_cbsa_cross_state",
        ],
        ordered=False
    )

    mismatch = pd.Series(pd.NA, index=out.index, dtype="object")

    mismatch = mismatch.mask(
        ~missing_mask & (same_cbsa == 1),
        "same_cbsa"
    )
    mismatch = mismatch.mask(
        ~missing_mask & (same_cbsa == 0) & (same_state == 1),
        "cross_cbsa_same_state"
    )
    mismatch = mismatch.mask(
        ~missing_mask & (same_state == 0),
        "cross_state"
    )

    out["hw_spatial_mismatch"] = pd.Categorical(
        mismatch,
        categories=[
            "same_cbsa",
            "cross_cbsa_same_state",
            "cross_state",
        ],
        ordered=True
    )

    out["hw_major_boundary"] = np.where(
        missing_mask,
        pd.NA,
        ((same_cbsa == 0) | (same_state == 0)).astype("int8")
    )

    combo = pd.Series(pd.NA, index=out.index, dtype="object")
    combo = combo.mask(
        ~missing_mask,
        "cbsa_" + same_cbsa.astype(str) + "_state_" + same_state.astype(str)
    )
    out["hw_cbsa_state_combo"] = combo

    return out