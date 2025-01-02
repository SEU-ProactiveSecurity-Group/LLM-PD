from enum import Enum
from dataclasses import dataclass


class DefenceStrategy(Enum):
    PORT_HOPPING = "PORT_HOPPING"
    REPLICA_INCREASE = "REPLICA_INCREASE"
    REPLICA_DECREASE = "REPLICA_DECREASE"
    REPLICA_EXPAND = "REPLICA_EXPAND"
    REPLICA_SHRINK = "REPLICA_SHRINK"
    NO_ACTION = "NO_ACTION"


class DeciderType(Enum):
    RANDOM = "RANDOM"
    GREEDY = "GREEDY"
    DQN = "DQN"
    AC = "AC"
    PPO = "PPO"
    LLM = "LLM"


class AttackerType(Enum):
    LDOS = "LDOS"
    DDOS = "DDOS"


@dataclass
class Indicators:
    C_e: float
    C_d: float
    M_e: float
    M_d: float
    con_delay: float
    mem_delay: float
    cost: int


map_action_to_defence = {
    0: DefenceStrategy.PORT_HOPPING,
    1: DefenceStrategy.REPLICA_INCREASE,
    2: DefenceStrategy.REPLICA_DECREASE,
    3: DefenceStrategy.REPLICA_EXPAND,
    4: DefenceStrategy.REPLICA_SHRINK,
    5: DefenceStrategy.NO_ACTION,
}


def check_attacker_type(attacker_type: str) -> AttackerType:
    if attacker_type not in AttackerType.__members__:
        raise ValueError(f"Invalid attacker type: {attacker_type}")
    return AttackerType.__members__[attacker_type]


def check_decider_type(decider_type: str) -> DeciderType:
    if decider_type not in DeciderType.__members__:
        raise ValueError(f"Invalid decider type: {decider_type}")
    return DeciderType.__members__[decider_type]
