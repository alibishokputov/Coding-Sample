import pandas as pd
import numpy as np

class DemographicFactors:
    
    @staticmethod
    def clean_hh_size(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return np.nan, np.nan
        if n < 1 or n > 12:
            return np.nan, np.nan
        n_continuous = min(n, 6)
        if n == 1:
            cat = "1_single"
        elif n == 2:
            cat = "2_dual"
        elif 3 <= n <= 4:
            cat = "3_small_family"
        else:
            cat = "4_large_family"
        return n_continuous, cat

    @staticmethod
    def clean_children_variable(row):
        kids = row["hh_children"]
        size = row["hh_size_numeric"]
        if pd.isna(kids) and size == 1:
            return 0.0
        if pd.notna(kids):
            try:
                kids = float(kids)
            except (ValueError, TypeError):
                return np.nan
            return min(kids, 4.0)
        return np.nan

    @staticmethod
    def clean_income_variable(val):
        s = str(val).lower().strip()
        if any(x in s for x in ['under_20k', 'less than $20,000', 'lt_20000', '<$20k',
                                '20000_34999', '$20,000 to $34,999', '20k_35k', '$20k-$35k',
                                '35000_49999', '$35,000 to $49,999', '35k-50k', '$35k-$50k']):
            return 'under_50k'
        elif any(x in s for x in ['50000_74999', '$50,000 to $74,999', '50k_75k', '$50k-$75k',
                                  '75000_99999', '$75,000 to $99,999', '75k_100k', '$75k-$100k']):
            return '50k_100k'
        elif any(x in s for x in ['100000_149999', '$100,000 to $149,999', '100k_150k', '$100k-$150k']):
            return '100k_150k'
        elif any(x in s for x in ['gt_150000', 'more than $150,000', 'above_150k', '>$150k']):
            return 'over_150k'
        return np.nan    

    @staticmethod
    def clean_education_variable(val):
        s = str(val).lower().strip()
        if s in ['master', 'doctoral', 'professional degree', 'phd']:
            return 'graduate_degree'
        elif s in ['bachelor', 'bachelors', 'bs', 'ba']:
            return 'bachelor_degree'
        elif s in ['high_school', 'high school', 'associate', 'some college', 'ged', 'no degree']:
            return 'pre_bachelor'
        return np.nan

    @staticmethod
    def clean_age_variable(val):
        s = str(val).strip()
        if s in ['18-24', '25-34']:
            return '18_34'
        elif s in ['35-44', '45-54']:
            return '35_54'      
        elif s in ['55-64', '65-74', '75+' ]:
            return '55_plus'    
        return np.nan

    @staticmethod
    def clean_gender_variable(val):
        s = str(val).lower().strip()        
        if s in ['woman', 'female']:
            return 'female'
        elif s in ['man', 'male']:
            return 'male'
        elif 'non' in s and 'binary' in s:
            return 'non_binary'
        return np.nan

    @staticmethod
    def clean_race_ethnicity_variable(row):
        race = row.get("race_ethnicity", np.nan)
        race_text = row.get("race_ethnicity_text", np.nan)
    
        if pd.isna(race) and pd.isna(race_text):
            return np.nan
    
        main_cat = str(race).lower().strip() if pd.notna(race) else ""
        text_entry = str(race_text).lower().strip() if pd.notna(race_text) else ""
        combined = f"{main_cat},{text_entry}".strip(",")
    
        # standardize separators a bit
        combined = combined.replace(";", ",")
        combined = combined.replace("|", ",")
        combined = combined.replace("native_hawaiian", "native hawaiian")
        combined = combined.replace("native_american", "native american")
        combined = combined.replace("middle_eastern", "middle eastern")
    
        if "prefer" in combined:
            return np.nan
    
        if any(x in combined for x in ["latinx", "latin", "hispanic"]):
            return "hispanic"
    
        tokens = [t.strip() for t in combined.split(",") if t.strip()]
        tokens = list(dict.fromkeys(tokens))  
    
        if not tokens:
            return np.nan
    
        if tokens == ["black"]:
            return "non_hispanic_black"
    
        if tokens == ["white"]:
            return "non_hispanic_white"
    
        if len(tokens) > 1:
            return "other_multiracial"
    
        if tokens[0] in ["asian", "native american", "native hawaiian", "middle eastern", "other"]:
            return "other_multiracial"
    
        return np.nan

    @staticmethod
    def clean_work_status_variable(val):
        s = str(val).lower().strip()
        if "full" in s:
            return "full_time"
        elif "part" in s:
            return "part_time"
        elif any(x in s for x in ["student", "multiple", "self"]):
            return "other_flexible"
        return np.nan

    @staticmethod
    def clean_vehicles_variable(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return np.nan
        if n == 0:
            return "0_car_free"
        elif n == 1:
            return "1_vehicle"
        elif n == 2:
            return "2_vehicles"
        elif n >= 3:
            return "3_plus_vehicles"
        return np.nan

    @staticmethod
    def household_composition(df):
        df = df.copy()
        df[["hh_size_numeric", "hh_size_cat"]] = df["hh_size"].apply(
            lambda x: pd.Series(DemographicFactors.clean_hh_size(x))
        )
        df["kids_clean_count"] = df.apply(
            DemographicFactors.clean_children_variable,
            axis=1
        )
        df["hh_size_clean_count"] = df["hh_size_numeric"]
        df["household_composition"] = "adults_shared"
        df.loc[df["hh_size_clean_count"] == 1, "household_composition"] = "solitary"
        df.loc[
            (df["hh_size_clean_count"] > 1) & (df["kids_clean_count"] > 0),
            "household_composition"
        ] = "family"
        return df
    
    @staticmethod
    def income_group(df):
        df = df.copy()
        df["income_clean"] = df["hh_income"].apply(DemographicFactors.clean_income_variable)
        return df

    @staticmethod
    def education_group(df):
        df = df.copy()
        df["education_clean"] = df["education"].apply(DemographicFactors.clean_education_variable)
        return df

    @staticmethod
    def age_group(df):
        df = df.copy()
        df["age_clean"] = df["age"].apply(DemographicFactors.clean_age_variable)
        return df

    @staticmethod
    def gender_group(df):
        df = df.copy()
        df["gender_clean"] = df["gender"].apply(DemographicFactors.clean_gender_variable)
        return df

    @staticmethod
    def race_group(df):
        df = df.copy()
        df["race_clean"] = df.apply(DemographicFactors.clean_race_ethnicity_variable, axis=1)
        return df

    @staticmethod
    def work_status_group(df):
        df = df.copy()
        df["work_status_clean"] = df["work_status"].apply(DemographicFactors.clean_work_status_variable)
        return df

    @staticmethod
    def vehicle_group(df):
        df = df.copy()
        df["hh_vehicles_clean"] = df["hh_vehicles"].apply(DemographicFactors.clean_vehicles_variable)
        return df

    @staticmethod
    def build_all(df):
        df = DemographicFactors.household_composition(df)
        df = DemographicFactors.income_group(df)
        df = DemographicFactors.education_group(df)
        df = DemographicFactors.age_group(df)
        df = DemographicFactors.gender_group(df)
        df = DemographicFactors.race_group(df)
        df = DemographicFactors.work_status_group(df)
        df = DemographicFactors.vehicle_group(df)
        return df



        
    
    
    
