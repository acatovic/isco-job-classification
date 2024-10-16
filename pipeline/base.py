import platform

def check_system_requirements() -> None:
    """
    Check if the system requirements are met.
    """
    assert platform.processor().lower() == "arm", "We only support Apple silicon, i.e. M1, M2, etc"
    assert platform.system().lower() == "darwin", "We only support MacOS"

def set_llama_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Llama expects the prompts to follow a particular format.

    Args:
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.

    Returns:
        str: The correctly formatted prompt.
    """
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
