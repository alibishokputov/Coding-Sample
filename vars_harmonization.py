import numpy as np

remote_potential_map = {
    "all": "full_remote_potential",
    "full_remote_potential": "full_remote_potential",
    "some": "partial_remote_potential",
    "partial_remote_potential": "partial_remote_potential",
    "none": "no_remote_potential",
    "no_remote_potential": "no_remote_potential",
}

remote_pref_map = {
    "always remote": "full_remote_prefer",
    "full_remote_prefer": "full_remote_prefer",
    "more remote": "partial_remote_prefer",
    "equally remote/in-person": "partial_remote_prefer",
    "more in-person": "partial_remote_prefer",
    "partial_remote_prefer": "partial_remote_prefer",
    "always in-person": "no_remote_prefer",
    "no_remote_prefer": "no_remote_prefer",
}

work_remote_3cat_map = {
    "in_person": "fully_in_person",
    "never": "fully_in_person",
    
    "hybrid": "hybrid",
    "sometimes": "hybrid",
    "infrequent_hybrid": "hybrid",
    "frequent_hybrid": "hybrid",
    
    "remote": "fully_remote",
    "always": "fully_remote",
    "almost always": "fully_remote",
    
    "multiple_worksites": np.nan,
}

work_remote_4cat_map = {
    "in_person": "fully_in_person",
    "never": "fully_in_person",
    
    "infrequent_hybrid": "infrequent_hybrid",
    
    "hybrid": "frequent_hybrid",
    "sometimes": "frequent_hybrid",
    "frequent_hybrid": "frequent_hybrid",
    
    "remote": "fully_remote",
    "always": "fully_remote",
    "almost always": "fully_remote",
    
    "multiple_worksites": np.nan,
}

work_remote_any_map = {
    "in_person": "no_remote_work",
    "never": "no_remote_work",
    
    "hybrid": "any_remote_work",
    "sometimes": "any_remote_work",
    "infrequent_hybrid": "any_remote_work",
    "frequent_hybrid": "any_remote_work",
    "remote": "any_remote_work",
    "always": "any_remote_work",
    "almost always": "any_remote_work",
    
    "multiple_worksites": np.nan,
}

work_remote_plus_mobile_map = {
    "in_person": "fully_in_person",
    "never": "fully_in_person",
    
    "hybrid": "hybrid",
    "sometimes": "hybrid",
    "infrequent_hybrid": "hybrid",
    "frequent_hybrid": "hybrid",
    
    "remote": "fully_remote",
    "always": "fully_remote",
    "almost always": "fully_remote",
    
    "multiple_worksites": "mobile_multi_site",
}