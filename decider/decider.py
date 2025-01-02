import decider.random as random_decider
import decider.llm as llm_decider
from constants import DeciderType


def deciderFactory(decider_type):
    if decider_type == DeciderType.LLM:
        return llm_decider
    elif decider_type == DeciderType.RANDOM:
        return random_decider
    else:
        raise ValueError("Invalid decider type")
