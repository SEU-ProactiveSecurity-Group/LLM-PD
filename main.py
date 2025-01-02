import argparse
from argparse import Namespace
from env import Env
from decider.decider import deciderFactory
from constants import (
    check_attacker_type,
    check_decider_type,
)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="argparse")

    # Add arguments
    parser.add_argument(
        "--num_episodes", type=int, required=False, default=5, help="Number of episodes"
    )
    parser.add_argument(
        "--max_fail_num", type=int, required=False, default=5, help="Max fail number"
    )
    parser.add_argument(
        "--max_episode_step",
        type=int,
        required=False,
        default=30,
        help="Max episode step",
    )

    parser.add_argument(
        "--attack_begin",
        type=bool,
        required=False,
        default=True,
        help="Attack from beginning",
    )
    parser.add_argument(
        "--attack_sequence",
        type=int,
        nargs="+",
        required=False,
        default=[10],
        help="Attack sequence",
    )

    parser.add_argument(
        "--decider_type", type=str, required=True, help="Type of the decider"
    )
    parser.add_argument(
        "--attacker_type", type=str, required=True, help="Type of the attacker"
    )
    parser.add_argument(
        "--attacker_num", type=int, required=True, help="Number of attackers"
    )
    parser.add_argument(
        "--change_num",
        type=int,
        required=False,
        default=0,
        help="Changed attacker num",
    )

    parser.add_argument(
        "--enable_log",
        type=bool,
        required=False,
        default=False,
        help="Enable log or not",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="default",
        help="Prefix of the log file",
    )

    args = parser.parse_args()

    # check the type of the attacker and the decider
    attacker_type = check_attacker_type(args.attacker_type)
    decider_type = check_decider_type(args.decider_type)

    # create the environment
    env_args = Namespace(
        attacker_type=attacker_type,
        attacker_num=args.attacker_num,
    )
    env = Env(env_args)

    prefix = args.prefix + "-" + args.decider_type

    # attack sequence
    attack_sequence_list = args.attack_sequence
    attack_sequence = []
    attack_begin = args.attack_begin
    for seq in attack_sequence_list:
        attack_sequence += [attack_begin] * seq
        attack_begin = not attack_begin

    # create the decider
    decider = deciderFactory(decider_type)
    decider.train_and_test(
        env=env,
        prefix=prefix,
        num_episodes=args.num_episodes,
        max_episode_step=args.max_episode_step,
        attack_sequence=attack_sequence,
        max_fail_num=args.max_fail_num,
        enable_log=args.enable_log,
        change_num=args.change_num,
    )
