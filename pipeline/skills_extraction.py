from typing import Any, Dict, List

from mlx_lm import generate

from base import set_llama_prompt

def get_job_description(s: str) -> str:
    """
    Extract the job description from the given string.

    Args:
        s (str): The string to extract the job description from.

    Returns:
        str: The job description.
    """
    return s.split("job_description:")[1].strip().lower()

def get_job_skills(s: str) -> List[str]:
    """
    Extract the job skills from the given string.

    Args:
        s (str): The string to extract the job skills from.

    Returns:
        List[str]: The job skills.
    """
    return [skill.strip() for skill in s.split("skills:")[1].strip().lower().split(",")]

def get_job_title(s: str) -> str:
    """
    Extract the job title from the given string.

    Args:
        s (str): The string to extract the job title from.

    Returns:
        str: The job title.
    """
    return s.split("job_title:")[1].strip().lower()

def get_parsed_job_dict(parsed_job_ad: str) -> Dict[str, Any]:
    """
    Parse the job description and extract the job title, job description, and job skills.

    Args:
        parsed_job_ad (str): The parsed job ad.

    Returns:
        Dict[str, Any]: The job title, job description, and job skills.
    """
    parsed_job_dict = {}
    for line in parsed_job_ad.split("\n"):
        if "job_title:" in line:
            parsed_job_dict["job_title"] = get_job_title(line)
        elif "job_description:" in line:
            parsed_job_dict["job_description"] = get_job_description(line)
        elif "skills:" in line:
            parsed_job_dict["skills"] = get_job_skills(line)
    return parsed_job_dict

def parse_job_ad(job_ad: str, model: Any, tokenizer: Any) -> str:
    """
    Parse the job ad and extract the job title, job description, and job skills.

    Args:
        job_ad (str): The job ad.

    Returns:
        str: The parsed job ad.
    """
    system_prompt = (
        "You are an expert at parsing online job ads.\n"
        "You are tasked with extracting the canonical job title, job description, and a list of job-specific skills, from a job ad.\n"
        "You are to use the following guidelines when extracting each of the aforementioned pieces of information:\n"
        "# JOB TITLE\n"
        "- Job title should be concise and typically specified by 1 to max 5 words;\n"
        "- Any marketing info, location info, or other superfluous information should be removed from the job title;\n"
        "- You are required to read the full job description thoroughly before concluding on the canonical job title;\n"
        "- Example 1: in the job ad beginning with 'User Researcher - Manchester...' the title is simply 'User Researcher';\n"
        "- Example 2: in the job ad beginning with 'Global Real Estate Private Equity Company - Financial Operations Manager (Cash Management and Treasury)...' the job title is simply 'Finance Manager'\n"
        "# JOB DESCRIPTION\n"
        "- Job description should be concise and stated by a single sentence;\n"
        "- Any marketing info and location info, should be removed from the job description;\n"
        "- Job description should ideally state what the job is about, and in which sector/industry;\n"
        "- Example 1: 'Panel and paint repairs in a well established accident repair centre'\n"
        "- Example 2: 'Providing patients care and consultations in a emergency department service';\n"
        "# SKILLS\n"
        "- Extracted job skills should be highly relevant and specific to this job, as presented in the job description;\n"
        "- Skills should typically consists of 2-5 words max, and should not contain certifications or qualifications;\n"
        "- Skills should be provided as a list, max 20 skills, seperated by a comma (',');\n"
        "- Some examples of skills for a 'legal policy officer': 'advise on legal decisions', 'compile legal documents', 'manage government policy implementation', etc;\n"
        "- Some examples of skills for a 'medical sales representative': 'medication classification', 'medical sales industry', 'advise on medical products', etc;\n"
        " The output of your job ad parsing should adhere to the following convention:\n"
        "job_title: <JOB_TITLE>\n"
        "job_description: <JOB_DESCRIPTION>\n"
        "skills: <SKILL_1>, <SKILL_2>, ..., <SKILL_n>\n"
    )
    skills_extraction_prompt = set_llama_prompt(system_prompt, job_ad)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=skills_extraction_prompt,
        max_tokens=4096,
    )
