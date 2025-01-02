# survival_rate and step_num
python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 50 --change_num 0 --enable_log True --prefix survival_rate

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type RANDOM --attacker_type LDOS --attacker_num 50 --change_num 0 --enable_log True --prefix survival_rate

# convergence_episode
python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 10 --change_num 0 --enable_log True --prefix convergence_episode

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 20 --change_num 0 --enable_log True --prefix convergence_episode

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 30 --change_num 0 --enable_log True --prefix convergence_episode

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 40 --change_num 0 --enable_log True --prefix convergence_episode

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 50 --change_num 0 --enable_log True --prefix convergence_episode

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type RANDOM --attacker_type LDOS --attacker_num 10 --change_num 0 --enable_log True --prefix convergence_episode

# migration_sucess_rate
python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type LLM --attacker_type LDOS --attacker_num 20 --change_num 50 --enable_log True --prefix migration_sucess_rate

python main.py --num_episodes 10 --max_fail_num 5 --attack_begin True --attack_sequence 10 --decider_type RANDOM --attacker_type LDOS --attacker_num 20 --change_num 50 --enable_log True --prefix migration_sucess_rate
