from .log import Logger
import os
import json


class LLMLogger(Logger):
    def __init__(
        self,
        prefix: str,
        title: str,
        log_dir: str = "log",
        txt_dir: str = "txt",
        prompts_dir: str = "prompts",
    ):
        super().__init__(prefix, title, log_dir, txt_dir)
        prompts_path = f"{self.dir_path}/{prefix}-{title}/{prompts_dir}"
        self.init_prompts(prompts_path)

    def close(self):
        super().close()
        self.close_prompts()

    # prompts
    def init_prompts(self, prompts_path: str):
        if not os.path.exists(prompts_path):
            os.makedirs(prompts_path)
        self.prompts_path = prompts_path

    def write_prompts(self, episode: int, prompts: list):
        prompts_path = f"{self.prompts_path}/{episode}.json"
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

    def close_prompts(self):
        pass
