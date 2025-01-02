from .ldos import LDoSAttacker
from constants import AttackerType


def attackerFactory(env, attacker_type, attacker_num=10):
    if attacker_type == AttackerType.LDOS:
        return LDoSAttacker(env, attacker_num)
    return None
