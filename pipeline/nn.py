from sentence_transformers import SentenceTransformer

QUERY_PROMPT_NAME = "s2p_query"

with open(OCCUPATION_DATA_PATH) as f:
    occupation_dict = json.load(f)

def preprocess_occupation_description(desc: str) -> str:
    desc = desc.lower()
    desc = desc.split("some related occupations classified elsewhere:")[0]
    desc = desc.split("excluded from this group are:")[0]
    desc = desc.split("\nnote")[0]
    return desc

occupation_id_to_text = {}

for id, data in occupation_dict.items():
    occupation_skills = "n/a"

    if "hasEssentialSkill" in data:
        occupation_skills = ", ".join(data["hasEssentialSkill"]).lower()

    occupation_titles = [data["title"]]

    if "alternativeLabel" in data["languages"]["en"]:
        occupation_titles += data["languages"]["en"]["alternativeLabel"]

    occupation_id_to_text[id] = (
        f"occupation titles: {', '.join(occupation_titles).lower()}; "
        f"occupation description: {preprocess_occupation_description(data['description'])}; "
        f"occupation skills: {occupation_skills}; "
    )

job_id_to_text = {}

for job_ad in parsed_job_ads:
    job_id_to_text[job_ad["competition_row_id"]] = (
        f"job title: {job_ad['job_title'].lower()}; "
        f"job description: {job_ad['job_description'].lower()}; "
        f"job skills: {', '.join(job_ad['skills']).lower()}; "
    )

def set_query_text(job_title: str, job_description: str, job_skills: List[str]) -> str:
    return (
        "We are looking for the closest job occupation category that matches the following data; "
        f"job title: {job_title}; "
        f"job description: {job_description}; "
        f"job skills: {', '.join(job_skills)}"
    ).lower()

job_ad_ids = []
query_texts = []

for job_ad in parsed_job_ads:
    job_ad_ids.append(job_ad["competition_row_id"])
    query_texts.append(
        set_query_text(
            job_ad["job_title"],
            job_ad["job_description"],
            job_ad["skills"]
        )
    )

def set_reference_text(job_titles: List[str], job_description: str, job_skills: List[str]) -> str:
    return (
        f"job titles: {', '.join(job_titles)}; "
        f"job_description: {job_description}; "
        f"job skills: {', '.join(job_skills)};"
    ).lower()



isco_codes = []  # short codes, i.e. 2422
esco_codes = []  # long codes, i.e. 2422.12 or 2422.12.4
reference_texts = []  # occupation titles, descriptions and skills as per ISCO/ESCO taxonomy

for esco_code, data in occupation_dict.items():
    isco_code = esco_code.split('.')[0]
    if isco_code != esco_code:  # exclude the l4 isco code info for now, only focus on lower levels, i.e. esco extension
        esco_codes.append(esco_code)
        isco_codes.append(isco_code)

        primary_job_title = data["languages"]["en"]["preferredLabel"]

        alternative_job_titles = []
        if "alternativeLabel" in data["languages"]["en"]:
            alternative_job_titles = data["languages"]["en"]["alternativeLabel"]

        job_titles = [primary_job_title] + alternative_job_titles
        job_description = data["description"]
        job_skills = data["hasEssentialSkill"]
        reference_texts.append(
            set_reference_text(
                job_titles=job_titles,
                job_description=job_description,
                job_skills=job_skills,
            )
        )

esco_codes = pd.Series(esco_codes)
isco_codes = pd.Series(isco_codes)

model = SentenceTransformer(
   "dunzhang/stella_en_400M_v5",
   trust_remote_code=True,
   device="cpu",
   config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)

query_embeddings = model.encode(query_texts, prompt_name=query_prompt_name)
doc_embeddings = model.encode(reference_texts)  # NOTE: LOAD EMBEDDINGS

sims = model.similarity(query_embeddings, doc_embeddings).numpy()


TOP_K = 5

nearest_topk_ixs = np.argsort(sims, axis=1)[:, -TOP_K:]

pred_codes = {}
job_ad_ids_for_llm_ranking = {}

for i, topk_ixs in enumerate(nearest_topk_ixs):
    codes = [isco_codes[topk_ix] for topk_ix in topk_ixs]
    c = Counter(codes)

    # rule 1: clear majority winner - if >=4 in top-5, then that is our prediction
    # rule 2: if top code AND >= 3 in top-5, then that is our prediction
    # all else are given to LLM to predict
    top_pred = c.most_common(n=1)[0]
    if top_pred[1] > 3:
        pred_codes[job_ad_ids[i]] = top_pred[0]
        continue

    if top_pred[1] == 3 and codes[-1] == top_pred[0]:
        pred_codes[job_ad_ids[i]] = top_pred[0]
        continue
    
    job_ad_ids_for_llm_ranking[job_ad_ids[i]] = (query_texts[i], [esco_codes[topk_ix] for topk_ix in topk_ixs])

    # Re-rank with LLM

system_prompt = """
**Instructions for Job Classification Using Chain-of-Thought Reasoning**

---

**Task Overview:**

Your task is to categorize a given job advertisement into one of the five ISCO occupation categories listed below. Each category includes potential job titles, a description, and a list of associated skills. You should carefully analyze the job advertisement and determine the best matching category based on the alignment of job titles, descriptions, and skills.

---

**Job Advertisement:**

- **Job Title:** [Insert Job Title]
- **Description:** [Insert Job Description]
- **Required Skills:** [Insert Required Skills]

---

**ISCO Occupation Categories:**

1. **Category 1**
   - **Potential Job Titles:** [List of Job Titles for Category 1]
   - **Description:** [Description of Category 1]
   - **Skills:** [List of Skills for Category 1]

2. **Category 2**
   - **Potential Job Titles:** [List of Job Titles for Category 2]
   - **Description:** [Description of Category 2]
   - **Skills:** [List of Skills for Category 2]

3. **Category 3**
   - **Potential Job Titles:** [List of Job Titles for Category 3]
   - **Description:** [Description of Category 3]
   - **Skills:** [List of Skills for Category 3]

4. **Category 4**
   - **Potential Job Titles:** [List of Job Titles for Category 4]
   - **Description:** [Description of Category 4]
   - **Skills:** [List of Skills for Category 4]

5. **Category 5**
   - **Potential Job Titles:** [List of Job Titles for Category 5]
   - **Description:** [Description of Category 5]
   - **Skills:** [List of Skills for Category 5]

---

**Instructions:**

1. **Read the Job Advertisement Carefully:**
   - Understand the job title, main responsibilities described, and the required skills.

2. **Analyze Each ISCO Category:**
   - For each category (1 to 5), do the following:
     - **Compare Job Titles:**
       - Check if the job title in the advertisement closely matches any of the potential job titles listed in the category.
     - **Compare Descriptions:**
       - Evaluate how well the job description aligns with the category's description.
     - **Compare Skills:**
       - Identify the overlap between the required skills in the advertisement and the skills listed for the category.

3. **Use Chain-of-Thought Reasoning:**
   - **Step-by-Step Analysis:**
     - Document your reasoning process as you compare the job advertisement with each category.
     - Note the strengths of the matches and any discrepancies.
   - **Example** (do not include in your final output):
     - "In Category 1, the job title matches, and most skills align, but the job description focuses on managerial duties not mentioned in Category 1."

4. **Determine the Best Matching Category:**
   - Based on your analysis, decide which category has the highest degree of alignment overall.
   - Consider the job title, description, and skills collectively.

5. **Final Output:**
   - Provide **only** the number corresponding to the best matching ISCO category (1, 2, 3, 4, or 5).
   - Do not include any additional text or explanation in the final output.

---

**Reminder:**

- Your analysis should be thorough and logical, but the final answer must be concise.
- The purpose of the Chain-of-Thought is to ensure accurate reasoning leading to the correct category.

---

**Final Answer Format:**

- Simply write the number of the chosen category.
- Example: `3`

---

**End of Instructions**
"""