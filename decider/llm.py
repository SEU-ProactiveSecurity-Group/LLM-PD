from tqdm import tqdm
from tenacity import retry, stop_after_attempt
import time
from openai import OpenAI
from pydantic import BaseModel
from dataclasses import asdict
from log.llm_log import LLMLogger
from utils import get_action_thresholds, judge_fail_func
import random


class Action(BaseModel):
    action: int
    # con_percent: float
    # mem_percent: float
    desc: str


class Judge(BaseModel):
    success: bool
    desc: str


class Reflex(BaseModel):
    desc: str


class LLM:
    def __init__(self, num_episodes, max_fail_num=5):
        self.client = OpenAI()
        self.inital_prompts = [
            {
                "role": "system",
                "content": """
                You are a security robot capable of continuously improving defense strategies across multiple "episodes" of DoS attacks. Each episode consists of multiple "steps" in the attack and defense processes. You must constantly monitor the service's "number of connections" and "memory usage" to ensure system security and service availability.
                In each step, the attacker may either launch an attack or remain stationary. If the attacker launches an attack, it will occupy a large number of replica connections or memory resources, resulting in an increase in the service connection load rate and memory load rate.
                As the defender, you need to perform two phases. In the "Decision" phase, you need to select one of six MTD strategies based on the current service status and evaluation indicators. After applying the defense, the "Judgment" phase begins, where the service status evaluation indicators are used to assess whether the defense was successful or failed.
                If the defense is successful over a specified number of consecutive steps, it indicates that the current defense strategy is effective against the attacker, and the episode ends. If the defense fails over a specified number of consecutive steps, it indicates that the defense strategy is not effective, and the episode ends.
                Between steps and episodes, "Step reflection" and "Episode reflection" are conducted to summarize the success and failure experiences:
                    Step reflection: Multiple attack and defense steps may form an action sequence with a mix of successes and failures. If the number of successes or failures is below a specified threshold, the entire episode will not be marked as successful or failed. However, if a sequence of consecutive failures occurs, particularly between the first success and the first failure, this sequence is considered a failure pattern. This failure sequence will be used in future step reflections to avoid repeating the same failed actions.
                    Episode reflection: After each episode, the overall success and failure experiences are summarized. If the episode was successful, the successful strategies will be summarized to guide decision-making in subsequent episodes. If the episode was a failure, the causes of failure will be analyzed, and different defense strategies will be applied in the next episode to avoid repeating the same failure sequence.
                A better strategy involves exploring different defense strategies during the initial episodes, ensuring a variety of action sequences while still maintaining successful defenses. This exploration minimizes the number of failed episodes and reduces resource consumption. In later episodes, the successful action sequences from previous episodes can be applied to improve decision-making and optimize defense strategies.
                """,
            },
            {
                "role": "system",
                "content": """
                In the decision phase, you need to assess the current service state to determine whether the service is in a critical condition and select a defense action, to either defend against the attacker's traffic or reduce resource consumption, ensuring the service operates normally.
                The service state, denoted as "state", is a 10x4 two-dimensional array representing the current service, which can have up to 10 "replicas". Each replica has four monitored status indicators: the number of "pods", the number of connections, the port number, and memory usage:
                    The number of replica pods (state[:][0]) is an integer between 0 and 100. A value of 0 indicates the replica does not exist; otherwise, the replica is active.
                    The number of connections (state[:][1]) is the total number of connections across all pods in the replica. Each pod can have between 0 and 256 connections, and a replica can have up to 100 pods, so the total number of connections ranges from 0 to 25600.
                    The port number (state[:][2]) is an integer between 30000 and 32767.
                    The memory usage (state[:][3]) is the total memory usage across all pods in the replica. Each pod can have memory usage between 0 and 100, and a replica can have up to 100 pods, so the total memory usage ranges from 0 to 10000.
                The defense action, denoted as "action", is an integer between 0 and 5, with each integer corresponding to a different MTD defense strategy. Some actions also include the connection load threshold "con_percent" and memory load threshold "mem_percent" parameters. Each action generates a certain resource consumption cost, denoted as "cost", defined as follows:
                    Action 0: Port hopping, which reassigns the replica's port numbers to clear all attacker connections and memory usage across all replicas. This action has a resource cost of cost = 4. It is prioritized to minimize resource consumption, but if other strategies fail, this action should be considered.
                    Action 1: Replica addition, which creates a copy of any replica with a connection load rate >= 0.8 or memory load rate >= 0.8. Half of the connections and memory usage are allocated to the new replica. This action is limited by the total number of replicas and has a resource cost of cost = 1.
                    Action 2: Replica removal, which deletes any replica with both connection load rate <= 0.3 and memory load rate <= 0.3. The connections and memory usage of the removed replica are redistributed to other replicas to improve resource utilization. The number of replicas cannot be less than 1, and the resource cost is cost = 0.
                    Action 3: Replica scaling, which increases the number of pods in any replica with a connection load rate >= 0.8 or memory load rate >= 0.8. After scaling, the connection load rate is reduced to below 0.8, or the memory load rate is reduced to below 0.8. This action is limited by the available number of pods and has a resource cost of cost = 1.
                    Action 4: Replica shrinking, which reduces the number of pods in any replica with a connection load rate <= 0.3 and memory load rate <= 0.3. After shrinking, the connection load rate stays above 0.3, and the memory load rate stays above 0.3, improving resource utilization. This action has a resource cost of cost = 0.
                    Action 5: No action, which maintains the current state without consuming resources. This action has a resource cost of cost = 0.
                "connection load rate" = replica connections / (replica pods * pod max connections). "memory load rate" = replica memory usage / (replica pods * pod max memory usage). Both load rates are float values between 0 and 1.
                In making a decision, you should first reason through the potential outcomes of applying a specific action, evaluating how the service state will change. Then, calculate the related indicators to check if the defense might fail or lead to excessive resource consumption. The action with the best match to the conditions should be selected. For example: 
                    If the service is in a critical state, such as the overall connection or memory load rates exceeding a specified threshold, action (Action 0, 1, 3) should be taken to defend against the attacker. Prioritize actions with lower resource costs while ensuring defense success.
                    If the service is in an inefficient state, with replica connection and memory load rates both below a certain threshold, action (Action 2, 4) should be taken to reduce resource utilization.
                    If the service is in a normal state, i.e., it does not meet the conditions for being in a critical or inefficient state, no action (Action 5) should be taken.
                """,
            },
            {
                "role": "system",
                "content": """
                In the judgment phase, the defender first executes the action. The execution may be successful or fail, for example, due to insufficient resources. 
                    If the execution is successful, it indicates that the selected action, con_percent, and mem_percent are valid. 
                    If the execution fails, it suggests that, under the current service state, choosing this action and parameters is ineffective, and a more appropriate action and parameters should be selected based on the service state next time.
                After executing the defense action, the service status evaluation indicators ("indicators") are obtained. You need to use these evaluation indicators to determine whether the defense of this step was successful, outputting the success status ("success") and the reason for failure ("desc").
                The evaluation indicators are a data structure containing seven metrics, as follows:
                    C_e: The inefficient service rate based on the number of connections (connection load rate < 0.3). It is calculated as C_e = Number of inefficient replicas / Total number of replicas.
                    C_d: The dangerous service rate based on the number of connections (connection load rate > 0.9). It is calculated as C_d = Number of dangerous replicas / Total number of replicas.
                    M_e: The inefficient service rate based on memory usage (memory load rate < 0.3). It is calculated as M_e = Number of inefficient replicas / Total number of replicas.
                    M_d: The dangerous service rate based on memory usage (memory load rate > 0.9). It is calculated as M_d = Number of dangerous replicas / Total number of replicas.
                    con_delay: The service delay based on the number of connections, i.e., the total number of service connections divided by the maximum possible connections for the service. In formula terms, con_delay = Sum of (replica pods * pod connections) across all replicas / Sum of (replica pods * pod max connections) across all replicas.
                    mem_delay: The service delay based on memory usage, i.e., the total memory usage of the service divided by the maximum possible memory usage for the service. In formula terms, mem_delay = Sum of (replica pods * pod memory usage) across all replicas / Sum of (replica pods * pod max memory usage) across all replicas.
                    cost: The resource consumption incurred by the defense action in the current step.
                The defense is considered a failure if any of the following conditions are met:
                    C_d > 0 or M_d > 0.8 or con_delay > 0.8 or mem_delay > 0.8.
                Resource utilization evaluation indicators:
                    While ensuring defense success, the objective is to minimize cost, C_e, and M_e, and also to keep con_delay and mem_delay as low as possible.
                """,
            },
            {
                "role": "system",
                "content": """
                In the Step reflection phase, the defender checks whether the current failed action sequence is a repetition of previously recorded failed action sequences. 
                    If it is not a repetition, the sequence is added to the list of failed action sequences. 
                    If it is a repetition, a warning is issued not to repeat the same failed actions and parameters in the current step.
                In the Episode reflection phase, the success information from the previous episode's attack-defense outcome, along with the action sequence "success_actions", is provided as input.
                The success_actions is a list containing multiple elements, each of which is an object that includes the "action", "defence_success" (whether the defense execution is successful), and "success" (whether the episode's defense is successful) attributes.
                    If the previous episode's defense was successful, the successful experience can be summarized to guide the defense strategy in the current step.
                    If the previous episode's defense failed, the cause of failure can be analyzed, and a different defense strategy can be adopted in the current step to avoid repeating the same failed action sequence.
                Additionally, in the earlier steps, the strategy should focus on exploration. During decision phase, actions should be chosen that differ from the sequences that were successful in previous steps, exploring whether there is a better action sequence to further reduce the number of failures in the current step. In later steps, successful action sequences from previous steps can be used to guide decision-making for the current step.
                """,
            },
        ]
        self.episode = 0
        self.num_episodes = num_episodes
        self.explore_x = -1
        self.explore_rate = 0
        self.explore_base = 0.5
        self.best_ep_actions = None
        self.take_best_action = False
        self.prompts = []
        self.actions = []
        self.defence_successes = []
        self.successes = []
        self.ep_fail_actions = []
        self.ep_success_actions = []
        self.fail_num = 0
        self.success_num = 0
        self.max_fail_num = max_fail_num
        self.step_actions = []
        self.step_fail_actions = []

    def reset(self):
        self.prompts = []
        self.fail_num = 0
        self.success_num = 0
        self.step_actions = []
        self.step_fail_actions = []
        self.take_best_action = False

    @retry(stop=stop_after_attempt(3))
    def take_action(self, state, attack_indicators, step, action_thresholds):
        print("action")

        best_actions = None
        if self.take_best_action:
            best_actions = self.best_ep_actions["actions"]
            if step >= len(best_actions):
                best_actions = None
        if best_actions:
            print("best actions", best_actions)

        prompts = [
            {
                "role": "user",
                "content": f"At the start of step {step}, the current defense service state is 'state': {str(state.tolist())}, and the action sequence taken in this step is 'cur_actions': {str(self.actions)}.",
            },
            {
                "role": "assistant",
                "content": f"[Decision] In the current service state state, predict whether the load rates have reached the danger threshold. Which defense action 'action' should the defender take to successfully defend against the attack in this step, while minimizing resource utilization indicators? Please provide an explanation for the choice 'desc'. {'The optimal action to consider for this step is ' + str(best_actions[step % len(best_actions)]) if best_actions else ''}.",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + self.prompts + prompts,
            response_format=Action,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed
        action = parsed.action
        con_percent, mem_percent = action_thresholds[action]
        self.actions.append(action)
        prompts += [
            {
                "role": "assistant",
                "content": f"The defense action to be taken in this step is {action}, with the connection load rate threshold set to {con_percent} and the memory usage threshold set to {mem_percent}. The reason for this decision is {parsed.desc}.",
            }
        ]
        self.prompts += prompts
        # return parsed.action, parsed.con_percent, parsed.mem_percent
        return action, con_percent, mem_percent

    def judge_fail(self, defence_state, defence_success, defence_fail_msg, indicators):
        print("judge_fail")
        success, fail_msg = judge_fail_func(indicators)
        self.defence_successes.append(defence_success)
        self.successes.append(success)

        if success:
            self.fail_num = 0
            self.success_num += 1
        else:
            self.success_num = 0
            self.fail_num += 1

        # whether the episode is finished
        finish = -1
        if self.fail_num >= self.max_fail_num:
            finish = 0
        if self.success_num >= self.max_fail_num:
            finish = 1

        prompts = [
            {
                "role": "user",
                "content": f"The defense action was {'successful' if defence_success else ('failed, reason: ' + defence_fail_msg) + ' and other actions may be needed in the next step.'}. The service state after execution is 'defence_state': {str(defence_state.tolist())}, and the evaluation indicators obtained are 'indicators': {asdict(indicators)}.",
            },
            {
                "role": "assistant",
                "content": "[Judgment] Was the defense successful or failed? If failed, what is the reason for failure?",
            },
        ]
        prompts += [
            {
                "role": "assistant",
                "content": f"The defense in this step was {'failed, reason for failure: ' + fail_msg if not success else 'successful'}.",
            }
        ]

        self.prompts += prompts
        return finish, success, fail_msg

    @retry(stop=stop_after_attempt(3))
    def reflex_step(self, action, step):
        some_steps_fail = self.success_num == 0
        repeated_fail_actions = True
        if len(self.step_fail_actions) == 0:
            repeated_fail_actions = False
        self.step_actions.append(action)
        if some_steps_fail:
            cur_actions = self.step_actions.copy()
            for a in self.step_fail_actions:
                if a != self.step_actions:
                    repeated_fail_actions = False
                    break
            if not repeated_fail_actions:
                self.step_fail_actions.append(cur_actions)
            self.step_actions = []

        if some_steps_fail:
            print("reflex certain steps")
            if repeated_fail_actions:
                prompt = f"[Step Reflection] The defenses in these steps ultimately ended in failure. The sequence of defense actions was {cur_actions}, which repeats a previous failed action sequence. All previous failed action sequences are {str(self.step_fail_actions)}. Be sure to avoid repeating the failed action sequences and make better defense decisions!"
            else:
                prompt = f"[Step Reflection] The defenses in these steps ultimately ended in failure. The sequence of defense actions was {cur_actions}. Please reflect on the failure experiences and try to improve the defense measures in the next rounds."

            prompts = [
                {
                    "role": "assistant",
                    "content": prompt,
                },
            ]

            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=self.inital_prompts + self.prompts + prompts,
                response_format=Reflex,
                timeout=30,
            )
            parsed = completion.choices[0].message.parsed
            print("step fail actions", cur_actions, self.step_fail_actions)
            self.prompts += prompts

    @retry(stop=stop_after_attempt(3))
    def reflex_ep(self, step_num, success, episode):
        print("reflex per episode")
        self.episode = episode
        fail_msg = (
            "The consecutive defense failures have reached the specified threshold!"
        )
        repeated_fail_actions = False
        repeated_success_actions = False
        if not success:
            for a in self.ep_fail_actions:
                if a["actions"] == self.actions:
                    repeated_fail_actions = True
                    break
            if not repeated_fail_actions:
                self.ep_fail_actions.append(
                    {"actions": self.actions.copy(), "fail_reason": fail_msg}
                )
        else:
            for a in self.ep_success_actions:
                if a["actions"] == self.actions:
                    repeated_success_actions = True
                    break
            if not repeated_success_actions:
                cur_ep_success_actions = {
                    "actions": self.actions.copy(),
                    "defence_successes": self.defence_successes,
                    "successes": self.successes,
                }
                self.ep_success_actions.append(cur_ep_success_actions)

        for a in self.ep_success_actions:
            if not self.best_ep_actions or len(a["actions"]) < len(
                self.best_ep_actions["actions"]
            ):
                self.best_ep_actions = a

        zip_ep_success_actions = [
            [
                {
                    "action": a["actions"][i],
                    "defence_success": a["defence_successes"][i],
                    "success": a["successes"][i],
                }
                for i in range(len(a["actions"]))
            ]
            for a in self.ep_success_actions
        ]

        x = (
            pow(self.explore_base, episode + self.explore_x)
            if (episode + self.explore_x) > 0
            else 0
        )
        self.explore_rate += x
        print("explore rate", episode, self.explore_rate)
        best_actions = None
        if self.best_ep_actions and (random.random() < self.explore_rate):
            best_actions = self.best_ep_actions["actions"]
            self.take_best_action = True
        else:
            self.take_best_action = False

        print("ep success actions", zip_ep_success_actions)

        prompts = [
            {
                "role": "user",
                "content": f"The attack-defense cycle in the previous episode has ended, with a total of {step_num} steps. The defense was {'successful' if success else ('failed, reason for failure: ' + fail_msg + '. The failed action sequence was: ' + str(self.actions))}.",
            },
            {
                "role": "user",
                "content": f"In the previous episode, { 'the failed actions from earlier episodes were repeated' if repeated_fail_actions else 'the failed actions from earlier episodes were not repeated' }. So far, the list of all failed defense actions across episodes is 'fail_actions': {str(self.ep_fail_actions)}. Please avoid repeating these actions in the current episode! Additionally, the list of all successful defense actions across episodes is 'success_actions': {str(zip_ep_success_actions)}.",
            },
            {
                "role": "assistant",
                "content": f"[Episode Reflection] After reflecting on the previous episode of the attack-defense process, we are currently in episode {episode}. {'In this episode, please plan a sequence of actions different from the previously successful action sequences and explore whether there is a better action sequence to achieve defense success in fewer steps.' if best_actions else ('In this episode, you can directly choose the optimal action sequence from previous episodes! The optimal action sequence to consider is: ' + str(best_actions))}.",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + self.prompts + prompts,
            response_format=Reflex,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed

        prompts += [
            {
                "role": "assistant",
                "content": f"Defense experience from the previous episode is:{parsed.desc}",
            }
        ]
        self.actions = []
        self.successes = []
        self.defence_successes = []
        self.prompts += prompts


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
        logger = LLMLogger(prefix, title)

    agent = LLM(num_episodes, max_fail_num)

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

        if episode != 0:
            agent.reflex_ep(for_step, for_episode_success, episode)
        elif change_num != 0 and episode == num_episodes - 1:
            env.change_attacker_num(change_num)

        with tqdm(total=max_steps, desc=f"iteration {episode}") as pbar:
            while finish == -1:
                print(f"\nstep {step}")
                do_attack = attack_sequence[step % attack_len]
                action_thresholds = get_action_thresholds(env.attacker.type)
                attack_indicators = env.cal_indicators(state)
                action, con_percent, mem_percent = agent.take_action(
                    state, attack_indicators, step, action_thresholds
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
                finish, success, fail_msg = agent.judge_fail(
                    defence_state, defence_success, defence_fail_msg, defence_indicators
                )
                print(
                    "defence_msg",
                    defence_success,
                    defence_fail_msg,
                    asdict(defence_indicators),
                )

                episode_success.append(success)

                agent.reflex_step(action, step)

                step += 1
                max_steps = max(max_steps, step)
                state = next_state

                if step >= max_episode_step:
                    finish = 0
                    success = 0
                    fail_msg = (
                        f"The defense was unsuccessful after {max_episode_step} steps!"
                    )

                print("reflex msg", finish, success, fail_msg)

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
            logger.write_prompts(episode, agent.prompts)

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
