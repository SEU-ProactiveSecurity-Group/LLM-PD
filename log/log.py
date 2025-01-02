import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


class Logger:
    def __init__(
        self,
        prefix: str,
        title: str,
        log_dir: str = "log",
        txt_dir: str = "txt",
    ):
        self.dir_path = f"{os.getcwd()}/output"
        log_path = f"{self.dir_path}/{prefix}-{title}/{log_dir}"
        txt_path = f"{self.dir_path}/{prefix}-{title}/{txt_dir}"
        self.init_log(log_path)
        self.init_txt(txt_path)

    def close(self):
        self.close_log()
        self.close_txt()

    # log
    def init_log(self, log_path: str):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path
        self.log_survival_rate = SummaryWriter(f"{log_path}")
        self.log_convergence_episode = SummaryWriter(f"{log_path}")
        self.log_success_list = SummaryWriter(f"{log_path}")
        self.log_step_num_list = SummaryWriter(f"{log_path}")

    def write_log(
        self,
        num_episodes: int,
        survival_rate: list,
        convergence_episode: int,
        success_list: list,
        step_num_list: list
    ):
        for i in range(num_episodes):
            self.log_survival_rate.add_scalar("survival_rate", survival_rate[i], i)
            self.log_success_list.add_scalar("success_list", success_list[i], i)
            self.log_step_num_list.add_scalar(
                "step_num_list", step_num_list[i], i
            )
        self.log_convergence_episode.add_scalar(
            "convergence_episode", convergence_episode
        )
        # save to csv
        df = pd.DataFrame(
            {
                "survival_rate": survival_rate,
                "success_list": success_list,
                "convergence_episode": convergence_episode,
                "step_num_list": step_num_list
            }
        )
        df.to_csv(f"{self.log_path}/log.csv", index=False)

    def close_log(self):
        self.log_survival_rate.close()
        self.log_convergence_episode.close()
        self.log_success_list.close()
        self.log_step_num_list.close()

    # txt
    def init_txt(self, txt_path: str):
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        self.txt_path = txt_path

    def write_txt(self, episode: int, txt_datas: list[dict]):
        for txt_data in txt_datas:
            for key, value in txt_data.items():
                file_path = os.path.join(self.txt_path, f"{key}.csv")
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"{value}\n")

    def close_txt(self):
        pass
