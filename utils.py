from constants import AttackerType


def get_action_thresholds(attacker_type: AttackerType):
    if attacker_type == AttackerType.DDOS:
        return [(0, 0), (0, 0.8), (0.3, 0.3), (0, 0.8), (0.3, 0.3), (0, 0)]
    else:
        return [(0, 0), (0.8, 0), (0.3, 0.3), (0.8, 0), (0.3, 0.3), (0, 0)]


def judge_fail_func(indicators):
    success, fail_msg = False, ""
    if indicators.C_d > 0:
        success = False
        fail_msg = "R_d > 0，There are replicas in a dangerous state, possibly because there are too many connections"
    elif indicators.M_d > 0:
        success = False
        fail_msg = "R_d > 0，There are replicas in a dangerous state, possibly because the memory usage is too high"
    elif indicators.con_delay > 0.8:
        success = False
        fail_msg = "con_delay > 0.8，The service delay is too high, possibly because there are too many connections"
    elif indicators.mem_delay > 0.8:
        success = False
        fail_msg = "mem_delay > 0.8，The service delay is too high, which may be caused by excessive memory usage."
    else:
        success = True
        fail_msg = None
    return success, fail_msg
