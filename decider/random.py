from tqdm import tqdm
import time
import random
from dataclasses import asdict
from log.log import Logger
from utils import judge_fail_func, get_action_thresholds


class Random:
    def __init__(self, max_fail_num=5):
        self.max_fail_num = max_fail_num
        self.fail_num = 0
        self.success_num = 0

    def reset(self):
        self.fail_num = 0
        self.success_num = 0

    def take_action(self, state, step, action_thresholds):
        action = random.randint(0, 5)
        con_threshold, mem_threshold = action_thresholds[action]
        return action, con_threshold, mem_threshold

    def judge(self, indicators):
        success, fail_msg = judge_fail_func(indicators)
        if success:
            self.fail_num = 0
            self.success_num += 1
        else:
            self.success_num = 0
            self.fail_num += 1

        finish = -1
        if self.fail_num >= self.max_fail_num:
            finish = 0
        if self.success_num >= self.max_fail_num:
            finish = 1
        return finish, success, fail_msg


def train_and_test(
    env,
    num_episodes,
    attack_sequence,
    max_fail_num,
    max_episode_step=30,
    enable_log=True,
    prefix="default",
    change_num=0,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    title = (
        env.attacker.type.value
        + "-"
        + str(env.attacker.num)
        + "-"
        + str(num_episodes)
        + "-"
        + str(change_num)
        + "-"
        + "("
        + timestamp
        + ")"
    )
    if enable_log:
        logger = Logger(prefix, title)

    agent = Random(max_fail_num)

    for_step = 0
    for_episode_success = False

    survival_rate = []
    convergence_episode = max_fail_num
    success_list = []
    step_num_list = []

    for episode in range(num_episodes):
        finish = -1
        step = 0
        attack_len = len(attack_sequence)
        max_steps = attack_len
        state = env.reset()
        agent.reset()

        txt_datas = []
        episode_success = []

        if change_num != 0 and episode == num_episodes - 1:
            env.change_attacker_num(change_num)

        with tqdm(total=max_steps, desc=f"iteration {episode}") as pbar:
            while finish == -1:
                print(f"\nstep {step}")
                do_attack = attack_sequence[step % attack_len]
                action_thresholds = get_action_thresholds(env.attacker.type)
                attack_indicators = env.cal_indicators(state)
                action, con_percent, mem_percent = agent.take_action(
                    state, step, action_thresholds
                )
                print(
                    "action_msg",
                    action,
                    con_percent,
                    mem_percent,
                    attack_indicators,
                )

                (
                    next_state,
                    defence_state,
                    defence_success,
                    defence_fail_msg,
                    defence_cost,
                ) = env.step(
                    action,
                    {"con_percent": con_percent, "mem_percent": mem_percent},
                    do_attack,
                )
                defence_indicators = env.cal_indicators(defence_state, defence_cost)
                finish, success, fail_msg = agent.judge(defence_indicators)
                print(
                    "defence_msg",
                    defence_success,
                    defence_fail_msg,
                    asdict(defence_indicators),
                )

                episode_success.append(success)

                step += 1
                max_steps = max(max_steps, step)
                state = next_state

                if step >= max_episode_step:
                    finish = 0
                    success = 0
                    fail_msg = (
                        f"The defense was unsuccessful after {max_episode_step} steps!"
                    )

                txt_datas.append(
                    {
                        "action": [action, con_percent, mem_percent],
                        "indicators": [
                            asdict(attack_indicators),
                            asdict(defence_indicators),
                        ],
                        "defence_msg": [defence_success, defence_fail_msg],
                        "success": [success, fail_msg],
                    }
                )

                pbar.set_postfix(
                    {
                        "episode": step,
                        "return": "%.3f" % (success / max_steps),
                    }
                )
                pbar.update(1)
                if max_steps != attack_len:
                    pbar.total = max_steps
                    pbar.refresh()

        for_step = step
        for_episode_success = finish == 1

        survival_rate.append(sum(episode_success) / len(episode_success))
        if step == max_fail_num and convergence_episode == max_fail_num:
            convergence_episode = episode
        success_list.append(for_episode_success)
        step_num_list.append(for_step)

        print(
            f"The {episode} episode has ended, with a total of {for_step} attack-defense cycles. The defense in this episode was {'successful' if for_episode_success else 'failed'}."
        )

        if enable_log:
            logger.write_txt(episode, txt_datas)

    if enable_log:
        logger.write_log(
            num_episodes,
            survival_rate,
            convergence_episode,
            success_list,
            step_num_list,
        )
        logger.close()

    return (
        survival_rate,
        convergence_episode,
        success_list,
        step_num_list,
    )
