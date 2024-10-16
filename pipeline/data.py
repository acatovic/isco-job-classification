import json
from typing import List

import pandas as pd

def load_job_ads(path: str) -> pd.DataFrame:
    """
    Load the job ads from the given path.
    """
    return pd.read_csv(path)

def load_occupations(path: str) -> pd.DataFrame:
    """
    Load the occupations from the given path.
    """
    with open(path, "r") as f:
        occupation_dict = json.load(f)

    isco_codes = []  # short codes, i.e. 2422
    esco_codes = []  # long codes, i.e. 2422.12 or 2422.12.4

    for esco_code, data in occupation_dict.items():
        isco_code = esco_code.split('.')[0]
        if isco_code == esco_code and occupation_dict[esco_code]["is_leaf"] == False:
            continue

        isco_codes.append(isco_code)
        esco_codes.append(esco_code)

    esco_codes = pd.Series(esco_codes)
    isco_codes = pd.Series(isco_codes)

    return (esco_codes, isco_codes, occupation_dict)

def preprocess_occupation_description(desc: str) -> str:
    """
    Preprocess the occupation description by removing unnecessary sections and converting to lowercase.

    Args:
        desc (str): The original occupation description.

    Returns:
        str: The preprocessed occupation description.
    """
    desc = desc.lower()
    desc = desc.split("some related occupations classified elsewhere:")[0]
    desc = desc.split("excluded from this group are:")[0]
    desc = desc.split("\nnote")[0]
    return desc
