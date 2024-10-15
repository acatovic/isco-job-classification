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
    pass