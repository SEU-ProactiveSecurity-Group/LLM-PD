import numpy as np
import math
from constants import DefenceStrategy


class Defender:
    def __init__(self, env):
        self.env = env

    def reset(self):
        """
        The simulation environment starts with 50 pods, which is half of the total pod count.
        The total number of connections is defined to be between 0.5 and 0.6 of the product of the total number of pods and 256, and the memory usage is between 10% and 50%.
        """
        self.env.ser_num = 5
        for i in range(self.env.ser_num):
            port = np.random.randint(30000, 32767)
            while port in self.env.state[:, 2]:
                port = np.random.randint(30000, 32767)
            connection = np.random.randint(int(10 * 256 * 0.5), int(10 * 256 * 0.6))
            mem = np.random.randint(int(10 * 100 * 0.1), int(10 * 100 * 0.5))
            self.env.state[i] = [10, connection, port, mem]

    def step(self, defence_strategy, params):
        inf_services = []
        if defence_strategy in [
            DefenceStrategy.REPLICA_INCREASE,
            DefenceStrategy.REPLICA_DECREASE,
            DefenceStrategy.REPLICA_EXPAND,
            DefenceStrategy.REPLICA_SHRINK,
        ]:
            con_percent = params["con_percent"]
            mem_percent = params["mem_percent"]
            if (
                con_percent < 0
                or con_percent > 1
                or mem_percent < 0
                or mem_percent > 1
                # or (con_percent == 0 and mem_percent == 0)
            ):
                return False, "The parameters are invalid.", 0
        if defence_strategy == DefenceStrategy.PORT_HOPPING:
            # Port hopping
            for i in range(self.env.ser_max_num):
                if self.env.state[i][0] > 0:
                    inf_services.append(i)
                    self.env.port_list.append(self.env.state[i][2])
                    # Reset the attack traffic while keeping the normal traffic unchanged.
                    if self.env.state[i][2] in self.env.attack_state[:, 0]:
                        ind = self.env.get_attack_index(self.env.state[i][2])
                        # Connection usage
                        self.env.state[i][1] = (
                            self.env.state[i][1] - self.env.attack_state[ind][4]
                        )
                        # Memory usage
                        self.env.state[i][3] = (
                            self.env.state[i][3] - self.env.attack_state[ind][5]
                        )
                    port = np.random.randint(30000, 32767)
                    while (
                        port in self.env.state[:, 2]
                    ):  # Make sure the port number does not overlap with the original port or other used ports.
                        port = np.random.randint(30000, 32767)
                    self.env.state[i][2] = port
            return (
                True,
                f"The port of service replica {inf_services} was changed successfully",
                4,
            )
        elif defence_strategy == DefenceStrategy.REPLICA_INCREASE:
            # Select service replicas with a load rate exceeding the specified threshold and create a new replica.
            # The number of pods for the new replica will be the same as the original service, and half of the total traffic will be assigned to the new replica.
            if self.env.ser_num == self.env.ser_max_num:
                return False, "The maximum number of replicas has been reached", 0
            elif self.env.pod_remain == 0:
                return False, "There are no remaining resource pods to allocate", 0
            else:
                for i in range(self.env.ser_max_num):
                    if (
                        self.env.state[i][1]
                        > con_percent * self.env.state[i][0] * self.env.pod_con_num
                        and self.env.state[i][3]
                        > mem_percent * self.env.state[i][0] * self.env.pod_mem_num
                    ):
                        if (
                            self.env.pod_remain >= self.env.state[i][0]
                            and self.env.ser_num < self.env.ser_max_num
                        ):
                            inf_services.append(i)
                            self.env.add_ser_list1.append(self.env.state[i][2])
                            new_pod = self.env.state[i][0]
                            connection = 0.5 * self.env.state[i][1]
                            self.env.state[i][1] = connection
                            mem = 0.5 * self.env.state[i][3]
                            self.env.state[i][3] = mem
                            port = np.random.randint(30000, 32767)
                            while port in self.env.state[:, 2]:
                                port = np.random.randint(30000, 32767)
                            for j in range(
                                self.env.ser_max_num
                            ):  # Locate the extended copy
                                if self.env.state[j][0] == 0:
                                    self.env.state[j] = np.array(
                                        [new_pod, connection, port, mem]
                                    )
                                    self.env.add_ser_list2.append(port)
                                    self.env.pod_remain -= new_pod
                                    self.env.ser_num += 1
                                    break
                        else:
                            return (
                                False,
                                "There are no remaining resource pods to allocate",
                                0,
                            )
                if len(inf_services) == 0:
                    return (
                        False,
                        "There are no replicas that exceed the specified load ratio, so there is no need to add replicas",
                        0,
                    )
                else:
                    return True, f"Replica added successfully {inf_services}", 1
        elif defence_strategy == DefenceStrategy.REPLICA_DECREASE:
            # Delete the replicas with a load rate lower than the specified one,
            # and distribute the traffic of the deleted services (including users and attackers) to other services;
            # The number of services cannot be less than 1
            all_delete = True
            for i in range(self.env.ser_max_num):
                if (
                    self.env.state[i][1]
                    >= con_percent * self.env.state[i][0] * self.env.pod_con_num
                    or self.env.state[i][3]
                    >= mem_percent * self.env.state[i][0] * self.env.pod_mem_num
                ):
                    all_delete = False
            if all_delete:
                return (
                    False,
                    "The load rate is set too high and all replicas will be deleted.",
                    0,
                )
            con_num = 0
            mem_num = 0
            for i in range(self.env.ser_max_num):
                if (
                    self.env.state[i][1]
                    < con_percent * self.env.state[i][0] * self.env.pod_con_num
                    and self.env.state[i][3]
                    < mem_percent * self.env.state[i][0] * self.env.pod_mem_num
                ):
                    inf_services.append(i)
                    con_num += self.env.state[i][1]
                    mem_num += self.env.state[i][3]
                    self.env.state[i] = np.array([0, 0, 0, 0])
                    self.env.ser_num -= 1
                    self.env.pod_remain += 1
            for j in range(self.env.ser_max_num):
                if self.env.state[j][0]:
                    self.env.state[j][1] += con_num // self.env.ser_num
                    self.env.state[j][3] += mem_num // self.env.ser_num
            if len(inf_services) == 0:
                return (
                    False,
                    "There are no replicas with a load ratio lower than the specified value, so there is no need to reduce the number of replicas.",
                    0,
                )
            else:
                return True, f"Replica deleted successfully {inf_services}", 0
        elif defence_strategy == DefenceStrategy.REPLICA_EXPAND:
            # Select all replicas with a load rate greater than the specified one and expand them.
            # After expansion, the load rate will be the specified load rate to ensure service quality;
            if self.env.pod_remain == 0:
                return False, "There are no remaining resource pods to allocate", 0
            else:
                for i in range(self.env.ser_max_num):
                    if (
                        self.env.state[i][1]
                        > con_percent * self.env.state[i][0] * self.env.pod_con_num
                        and self.env.state[i][3]
                        > mem_percent * self.env.state[i][0] * self.env.pod_mem_num
                    ):
                        con_incre = (
                            int(
                                math.ceil(
                                    self.env.state[i][1]
                                    / (self.env.pod_con_num * con_percent)
                                    - self.env.state[i][0]
                                )
                            )
                            if con_percent != 0
                            else 0
                        )
                        mem_incre = (
                            int(
                                math.ceil(
                                    self.env.state[i][3]
                                    / (self.env.pod_mem_num * mem_percent)
                                    - self.env.state[i][0]
                                )
                            )
                            if mem_percent != 0
                            else 0
                        )
                        pod_incre = max(con_incre, mem_incre)
                        if self.env.pod_remain >= pod_incre:
                            inf_services.append(i)
                            self.env.state[i][0] = self.env.state[i][0] + pod_incre
                            self.env.pod_remain -= pod_incre
                        else:
                            return (
                                False,
                                "There are no remaining resource pods to allocate",
                                0,
                            )
                if len(inf_services) == 0:
                    return (
                        False,
                        "There are no replicas that exceed the specified load ratio, so there is no need to add resource pods",
                        0,
                    )
                else:
                    return True, f"服务{inf_services}副本扩容成功", 1
        elif defence_strategy == DefenceStrategy.REPLICA_SHRINK:
            # Select replicas with a load rate lower than the specified load rate and scale them down.
            # The load rate after scaling down is the specified load rate to ensure the lowest energy consumption ratio;
            for i in range(self.env.ser_max_num):
                if (
                    self.env.state[i][1]
                    < con_percent * self.env.state[i][0] * self.env.pod_con_num
                    and self.env.state[i][3]
                    < mem_percent * self.env.state[i][0] * self.env.pod_mem_num
                ):
                    con_decre = int(
                        self.env.state[i][0]
                        - self.env.state[i][1] / (self.env.pod_con_num * con_percent)
                    )
                    mem_decre = int(
                        self.env.state[i][0]
                        - self.env.state[i][3] / (self.env.pod_mem_num * mem_percent)
                    )
                    pod_decre = max(con_decre, mem_decre)
                    inf_services.append(i)
                    self.env.state[i][0] = self.env.state[i][0] - pod_decre
                    self.env.pod_remain += pod_decre
            if len(inf_services) == 0:
                return (
                    False,
                    "There are no replicas with a load ratio lower than the specified one, so no scaling down is required.",
                    0,
                )
            else:
                return (
                    True,
                    f"The replica was successfully scaled down {inf_services}",
                    0,
                )
        elif defence_strategy == DefenceStrategy.NO_ACTION:
            return True, "No action", 0

        return True, None, 0
