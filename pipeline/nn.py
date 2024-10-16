import pickle
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_PATH, OCCUPATIONS_EMBEDDINGS_PATH

QUERY_PROMPT_NAME = "s2p_query"

def set_query_text(job_title: str, job_description: str, job_skills: List[str]) -> str:
    """
    Set the query text for job matching.

    Args:
        job_title (str): The title of the job.
        job_description (str): A description of the job.
        job_skills (List[str]): A list of skills required for the job.

    Returns:
        str: A formatted, lowercase string containing the job information for querying.
    """
    return (
        "We are looking for the closest job occupation category that matches the following data; "
        f"job title: {job_title}; "
        f"job description: {job_description}; "
        f"job skills: {', '.join(job_skills)}"
    ).lower()

def prepare_queries(parsed_job_ads: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Prepare query texts for job matching based on parsed job advertisements.

    This function takes a list of parsed job advertisements and generates query texts
    for each job ad. It uses the set_query_text function to format the job information
    into a standardized query string.

    Args:
        parsed_job_ads (List[dict]): A list of dictionaries, where each dictionary
                                     represents a parsed job advertisement containing
                                     'id', and 'title_and_description' keys.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - job_ad_ids: A list of job advertisement IDs (competition_row_id).
            - query_texts: A list of formatted query texts corresponding to each job ad.

    Note:
        This function assumes that the set_query_text function is defined and available
        in the current scope.
    """
    job_ad_ids = []
    query_texts = []

    for job_ad in parsed_job_ads:
        job_ad_ids.append(job_ad["id"])
        query_texts.append(set_query_text(job_ad["job_title"], job_ad["job_description"], job_ad["skills"]))
    
    return (job_ad_ids, query_texts)

def set_reference_text(job_titles: List[str], job_description: str, job_skills: List[str]) -> str:
    return (
        f"job titles: {', '.join(job_titles)}; "
        f"job_description: {job_description}; "
        f"job skills: {', '.join(job_skills)};"
    ).lower()

def nn(query_texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
        device="cpu",
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
    )

    with open(OCCUPATIONS_EMBEDDINGS_PATH, "rb") as f:
        occupations_embs = pickle.load(f)

    query_embeddings = model.encode(query_texts, prompt_name=QUERY_PROMPT_NAME)

    return model.similarity(query_embeddings, occupations_embs).numpy()
