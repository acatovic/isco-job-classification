import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

from mlx_lm import load
import numpy as np
import pandas as pd

from base import check_system_requirements
from config import LLAMA_MODEL_PATH
from data import load_job_ads, load_occupations
from nn import nn, prepare_queries
from skills_extraction import get_parsed_job_dict, parse_job_ad
from translation import translate_to_english

check_system_requirements()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model, tokenizer = load(LLAMA_MODEL_PATH)

def translation_pipeline(job_ads_path: str, output_dir: str) -> None:
    """
    Translate the job ads to English. Output the results to a CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_path = output_path / "job_ads_translated.csv"

    logger.info("Starting translation pipeline")
    logger.info(f"Translating job ads to English and saving to {output_path}")

    df = load_job_ads(job_ads_path)
    df["title_n_description"] = df[["title", "description"]].agg("; ".join, axis=1)
    df["title_n_description_en"] = df["title_n_description"].apply(translate_to_english, model=model, tokenizer=tokenizer)

    pd.DataFrame([
        pd.Series(df["id"], name="id").astype(int),
        pd.Series(df["title_n_description_en"], name="title_and_description").astype(str),
    ]).T.to_csv(output_path, index=False, header=True)

def parsing_pipeline(output_dir: str) -> None:
    """
    Parse the job ads. Output the results to a CSV file.
    """
    df = load_job_ads(Path(output_dir) / "job_ads_translated.csv")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_path = output_path / "job_ads_parsed.csv"

    parsed_job_dicts = []

    logger.info("Starting parsing pipeline")
    logger.info(f"Parsing job ads and saving to {output_path}")

    for ix, row in df.iterrows():
        parsed_job_ad = parse_job_ad(row["title_and_description"], model=model, tokenizer=tokenizer)
        parsed_dict = get_parsed_job_dict(parsed_job_ad)
        parsed_dict["id"] = row["id"]
        parsed_job_dicts.append(parsed_dict)

    with open(output_path, "w") as f:
        json.dump(parsed_job_dicts, f)

def nn_pipeline(output_dir: str) -> Tuple[List[int], np.ndarray]:
    """
    Run the nearest neighbor pipeline. Output the results to a CSV file.
    """
    path = Path(output_dir) / "job_ads_parsed.csv"
    with open(path, "r") as f:
        parsed_job_ads = json.load(f)
    
    job_ad_ids, query_texts = prepare_queries(parsed_job_ads)

    return (job_ad_ids, nn(query_texts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the job ads CSV file")
    parser.add_argument("--occupations", type=str, required=True, help="Path to the occupations JSON file")
    parser.add_argument("--output", type=str, required=False, default="../output/", help="Output directory")
    args = parser.parse_args()

    #translation_pipeline(args.data, args.output)

    #parsing_pipeline(args.output)

    #with open("../embeddings/stella_400m_occupations_embs.pkl", "rb") as f:
    #    occupations_embs = pickle.load(f)

    #print(type(occupations_embs))
    #print(occupations_embs.shape)

    esco_codes, isco_codes, occupation_dict = load_occupations(args.occupations)
    job_ad_ids, sims = nn_pipeline(args.output)
    print(sims)
    
