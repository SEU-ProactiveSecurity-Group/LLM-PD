import gymnasium as gym
from gymnasium import spaces
import numpy as np
from attacker.attacker import attackerFactory
from defender.defender import Defender
from constants import Indicators, map_action_to_defence


class Env(gym.Env):
    def __init__(self, args):
        self.pod_max_num = 100  # The total number of resource pods of the service
        self.pod_con_num = 256  # Maximum number of connections per pod
        self.pod_mem_num = 100  # Maximum memory usage of a single pod
        self.ser_max_num = 10  # Maximum number of replicas
        self.ser_ind = 4  # The number of indicators of the replicas
        self.ser_num = 0  # Current number of replicas
        self.con_danger_thresh_percent = 0.9  # Dangerous service connection threshold
        self.con_effective_thresh_percent = (
            0.3  # Inefficient service connection threshold
        )
        self.mem_danger_thresh_percent = 0.9  # Dangerous service memory usage threshold
        self.mem_effective_thresh_percent = (
            0.1  # Inefficient service memory usage threshold
        )

        # status indicators of each replica: number of pods, number of connections, port number, memory usage
        high = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        low = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        for i in range(self.ser_max_num):
            high[i] = [100, 25600, 32767, 10000]
            low[i] = [0, 0, 30000, 0]

        self.observation_space = spaces.Box(
            low, high, shape=(self.ser_max_num, self.ser_ind), dtype=np.int64
        )  # Box（10，4）

        self.defence_num = 6
        self.action_space = spaces.Discrete(
            self.defence_num
        )  # The size of the action space, one dimension

        self.attacker = attackerFactory(self, args.attacker_type, args.attacker_num)
        self.defender = Defender(self)
        self.defence_strategy = None

    def reset(self):
        self.state = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        self.attack_state = np.zeros((self.ser_max_num, 6), dtype=np.int64)
        self.steps_beyond_terminated = 0

        self.defender.reset()
        self.attacker.reset()

        return np.array(self.state, dtype=np.int64)

    def step(self, action, params, do_attack=True, simulate=False):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self.pod_remain = self.pod_max_num - np.sum(
            self.state[:, 0]
        )  # The number of remaining pods is the computing resources
        self.port_list = (
            []
        )  # Record the original port of the service that changes the port after the attacker attacks
        self.add_ser_list1 = []  # replica service to expand
        self.add_ser_list2 = []  # New services generated by extended replicas
        self.del_ser_list = []  # deleted replica service

        save_state = self.state.copy()  # Save the service status of the previous moment
        save_attack_state = (
            self.attack_state.copy()
        )  # Save the attack status of the previous moment
        save_ser_num = (
            self.ser_num
        )  # Save the number of services at the previous moment

        # transfer action to defence_strategy
        defence_strategy = map_action_to_defence[action]
        defence_success, defence_fail_msg, defence_cost = self.defender.step(
            defence_strategy, params
        )  # execute defence strategy

        defence_state = self.state.copy()  # The service status after defense

        # The current disadvantage is that do_attack actually determines whether to attack in the next round, not the current round.
        if do_attack:
            self.attacker.step(
                defence_strategy, simulate
            )  # Input attack traffic and execute attack strategy according to defense strategy
        else:
            self.state = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
            self.attacker.reset()  # Silent for one round, no attack
            self.defender.reset()  # Normal user traffic, clear attack traffic
        next_state = self.state.copy()  # The service status at the next moment

        # If it is a simulated execution, then the defense is executed and restored to the previous state
        if simulate:
            self.state = save_state
            self.attack_state = save_attack_state
            self.ser_num = save_ser_num

        return (
            next_state,
            defence_state,
            defence_success,
            defence_fail_msg,
            defence_cost,
        )

    def change_attacker_num(self, num):
        if self.attacker.num == num:
            return
        self.attacker = attackerFactory(self, self.attacker.type, num)
        self.reset()

    def cal_indicators(self, state, cost=0):
        con_effective_flag = 0  # Number of inefficient services
        con_danger_flag = 0  # Number of dangerous services
        mem_effective_flag = 0
        mem_danger_flag = 0
        pod_num = 0  # The number of pods for the service
        pod_con_num = 0  # Number of connections for the service
        pod_mem_num = 0  # Memory usage of the service
        for i in range(self.ser_max_num):
            if state[i][0] > 0:
                pod_num += state[i][0]
                pod_con_num += state[i][1]
                pod_mem_num += state[i][3]
                # con
                if (
                    state[i][1]
                    > state[i][0] * self.pod_con_num * self.con_danger_thresh_percent
                ):
                    con_danger_flag += 1
                elif (
                    state[i][1]
                    < self.con_effective_thresh_percent * state[i][0] * self.pod_con_num
                ):
                    con_effective_flag += 1
                # mem
                if (
                    state[i][3]
                    > state[i][0] * self.pod_mem_num * self.mem_danger_thresh_percent
                ):
                    mem_danger_flag += 1
                elif (
                    state[i][3]
                    < self.mem_effective_thresh_percent * state[i][0] * self.pod_mem_num
                ):
                    mem_effective_flag += 1
        C_e = con_effective_flag / self.ser_num  # Proportion of inefficient services
        C_d = con_danger_flag / self.ser_num  # Proportion of dangerous services
        M_e = mem_effective_flag / self.ser_num
        M_d = mem_danger_flag / self.ser_num
        con_delay = pod_con_num / (pod_num * self.pod_con_num)  # Service delay
        mem_delay = pod_mem_num / (pod_num * self.pod_mem_num)
        indicators = Indicators(C_e, C_d, M_e, M_d, con_delay, mem_delay, cost)
        return indicators

    def cal_reward(
        self, success, defence_success, defence_indicators, success_num, fail_num
    ):
        alpha, beta, gamma, delta = 10, 1, 1, 5
        success_flag = 1 if success else -1
        defence_success_flag = 1 if defence_success else -1
        time_cost = 2
        reward = (
            alpha * success_flag
            + beta * (defence_success_flag + success_num - fail_num)
            - gamma * (defence_indicators.cost + time_cost)
            - delta * (defence_indicators.M_d + defence_indicators.C_d)
        )
        return reward

    def get_state_index(self, port):
        return self.state[:, 2].tolist().index(port)

    def get_attack_index(self, port):
        return self.attack_state[:, 0].tolist().index(port)
