import numpy as np
from constants import AttackerType, DefenceStrategy


class LDoSAttacker:
    def __init__(self, env, num):
        self.num = num
        self.con_ab = 256  # The attacker's connection occupation capability
        self.mem_ab = 30  # Attacker's memory occupation capability
        self.env = env
        self.type = AttackerType.LDOS

    def reset(self):
        # The attacker can observe the information matrix in the environment:
        # service port number, load rate (connection or memory load), number of attacks, attack weight, attack traffic, memory usage
        self.con_ability = self.num * self.con_ab
        self.con_remain = self.con_ability
        self.mem_ability = self.num * self.mem_ab
        self.mem_remain = self.mem_ability

    def step(self, defence_strategy, simulate=False):
        save_con_remain = self.con_remain
        save_mem_remain = self.mem_remain

        # Changes in attacker traffic after defense actions
        if (
            defence_strategy == DefenceStrategy.PORT_HOPPING
        ):  # Service attack traffic that has port changes should be recovered
            for port in self.env.port_list:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    self.con_remain += self.env.attack_state[ind][4]
                    self.mem_remain += self.env.attack_state[ind][5]
                    self.env.attack_state[ind][4] = 0
                    self.env.attack_state[ind][5] = 0
        elif (
            defence_strategy == DefenceStrategy.REPLICA_INCREASE
        ):  # Add a replica, the attack traffic needs to be allocated to half of the new replica, and a new service needs to be added in attack_state
            for port in self.env.add_ser_list1:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    con_tmp = 0.5 * self.env.attack_state[ind][4]
                    mem_tmp = 0.5 * self.env.attack_state[ind][5]
                    self.env.attack_state[ind][4] = con_tmp
                    self.env.attack_state[ind][5] = mem_tmp
                    new_port = self.env.add_ser_list2[
                        self.env.add_ser_list1.index(port)
                    ]
                    for i in range(self.env.ser_max_num):
                        if self.env.attack_state[i][0] == 0:
                            self.env.attack_state[i][0] = new_port
                            self.env.attack_state[i][4] = con_tmp
                            self.env.attack_state[i][5] = mem_tmp
                            break
        elif defence_strategy == DefenceStrategy.REPLICA_DECREASE:
            for port in self.env.del_ser_list:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    attack_con = self.env.attack_state[ind][4]
                    attack_mem = self.env.attack_state[ind][5]
                    self.env.attack_state[ind][4] = 0
                    self.env.attack_state[ind][5] = 0
                    for i in range(self.env.ser_max_num):
                        if self.env.attack_state[i][0]:
                            self.env.attack_state[i][4] += (
                                attack_con // self.env.ser_num
                            )
                            self.env.attack_state[i][5] += (
                                attack_mem // self.env.ser_num
                            )
                            break

        # Reconnaissance phase: The attacker builds an observation matrix in the first round, and then only needs to add or delete ports and corresponding services;
        # the defender performs port transformation, and the attacker remains silent for a round and does not attack
        if self.env.port_list == []:
            # Determine the latency and other parameters based on the port.
            for port in self.env.attack_state[
                :, 0
            ]:  # First delete the ports that no longer exist in the state and assign them all to 0
                if port not in self.env.state[:, 2]:
                    ind = self.env.get_attack_index(port)
                    self.env.attack_state[ind] = np.array([0, 0, 0, 0, 0, 0])
            for port in self.env.state[
                :, 2
            ]:  # Add the newly added service in the state: only modify the port number, delay, and weight
                ind_s = self.env.get_state_index(port)
                if port > 0:
                    if port in self.env.attack_state[:, 0]:
                        ind = self.env.get_attack_index(port)
                        self.env.attack_state[ind][0] = self.env.state[ind_s][
                            2
                        ]  # Service port number detected by the attacker
                        # The service latency is expressed by dividing the number of service connections by the number of connections that the service can carry.
                        # The latency is too small to be reflected after rounding, so it is increased by 100 times.
                        self.env.attack_state[ind][1] = (
                            100
                            * self.env.state[ind_s][1]
                            / (self.env.state[ind_s][0] * self.env.pod_con_num)
                        )
                        # Calculating weights by delay
                        self.env.attack_state[ind][3] = 0.9 * self.env.attack_state[
                            ind
                        ][1] + 0.1 * 100 * (
                            self.env.attack_state[ind][2]
                            / (self.env.steps_beyond_terminated + 1)
                        )
                    else:
                        for i in range(self.env.ser_max_num):
                            if self.env.attack_state[i][0] == 0:
                                self.env.attack_state[i][0] = self.env.state[ind_s][2]
                                self.env.attack_state[i][1] = (
                                    100
                                    * self.env.state[ind_s][1]
                                    / (self.env.state[ind_s][0] * self.env.pod_con_num)
                                )
                                self.env.attack_state[i][
                                    3
                                ] = 0.9 * self.env.attack_state[i][1] + 0.1 * 100 * (
                                    self.env.attack_state[i][2]
                                    / (self.env.steps_beyond_terminated + 1)
                                )
                                break

            # Attack target selection
            target_list = []
            for port in self.env.attack_state[:, 0]:
                if port > 0:
                    target_list.append(port)

            # Start the attack and distribute the attack traffic according to the port
            if self.con_remain > 0:
                for port in target_list:
                    # Find the attacked service number in state, because state and attack_state are connected through port
                    target_ser_num = self.env.get_state_index(port)
                    target = self.env.get_attack_index(port)
                    # Number of attacks
                    self.env.attack_state[target][2] += 1
                    # Number of attack connections
                    attack_con = (
                        self.con_remain
                        * self.env.attack_state[target][3]
                        // np.sum(self.env.attack_state[:, 3])
                    )
                    if attack_con <= (
                        self.env.state[target_ser_num][0] * self.env.pod_con_num
                        - self.env.state[target_ser_num][1]
                    ):
                        self.env.state[target_ser_num][1] += attack_con
                        self.env.attack_state[target][4] += attack_con
                        self.con_remain -= attack_con
                    else:
                        self.env.attack_state[target][4] += (
                            self.env.state[target_ser_num][0] * self.env.pod_con_num
                            - self.env.state[target_ser_num][1]
                        )
                        self.con_remain -= (
                            self.env.state[target_ser_num][0] * self.env.pod_con_num
                            - self.env.state[target_ser_num][1]
                        )
                        self.env.state[target_ser_num][1] = (
                            self.env.state[target_ser_num][0] * self.env.pod_con_num
                        )  # Fully load the attacked service
                    # Attack memory usage
                    attack_mem = (
                        self.mem_remain
                        * self.env.attack_state[target][3]
                        // np.sum(self.env.attack_state[:, 3])
                    )
                    if attack_mem <= (
                        self.env.state[target_ser_num][0] * self.env.pod_mem_num
                        - self.env.state[target_ser_num][3]
                    ):
                        self.env.state[target_ser_num][3] += attack_mem
                        self.env.attack_state[target][5] += attack_mem
                        self.mem_remain -= attack_mem
                    else:
                        self.env.attack_state[target][5] += (
                            self.env.state[target_ser_num][0] * self.env.pod_mem_num
                            - self.env.state[target_ser_num][3]
                        )
                        self.mem_remain -= (
                            self.env.state[target_ser_num][0] * self.env.pod_mem_num
                            - self.env.state[target_ser_num][3]
                        )
                        self.env.state[target_ser_num][3] = (
                            self.env.state[target_ser_num][0] * self.env.pod_mem_num
                        )

        # If it is a simulated execution, then the defense is executed and restored to the previous state
        if simulate:
            self.con_remain = save_con_remain
            self.mem_remain = save_mem_remain
