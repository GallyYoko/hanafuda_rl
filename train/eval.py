from tqdm import tqdm

from hanafuda_rl.envs.hanafuda_env import HanafudaEnv
from hanafuda_rl.agents.random_agent import RandomAgent
from hanafuda_rl.agents.rule_agent import RuleAgent
from hanafuda_rl.agents.sb3_agent import PPOAgent 

# 玩家0 (主要评估对象)
AGENT_0_TYPE = 'ppo'
AGENT_0_PATH = "Hanafuda-Project/hanafuda_rl/results/models/hanafuda_ppo_selfplay_5M.zip"

# 玩家1 (对手)
AGENT_1_TYPE = 'ppo'
AGENT_1_PATH = "Hanafuda-Project/hanafuda_rl/results/models/hanafuda_ppo_selfplay_5M.zip"

NUM_GAMES = 10000
SEED = 99

def evaluate_duel(agent0, agent1, num_games=1000, seed=None):
    """
    公平地评估两个智能体。
    1. 环境内部 (`reset`) 会随机决定哪个玩家ID (0或1) 先手。
    2. 此函数通过明确的 `player_mapping` 确保 agent0 和 agent1
       在整个评估中被分配为玩家0和玩家1的次数相等。
    """

    env = HanafudaEnv()
    
    stats = {"wins_agent0": 0, "wins_agent1": 0, "draws": 0, "total_score_agent0": 0}

    for i in tqdm(range(num_games), desc="Evaluating Games"):
        player_mapping = {0: agent0, 1: agent1} # agent0 是 P0, agent1 是 P1

        obs, _ = env.reset(seed=seed + i if seed is not None else None)
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            current_player_id = env.unwrapped.current_player
            action_mask = env.unwrapped.get_action_mask()
            
            # 使用映射来找到当前应该行动的智能体
            active_agent = player_mapping[current_player_id]
            action = active_agent.select_action(obs, action_mask)
            
            obs, _, terminated, truncated, _ = env.step(action)

        # 记录结果
        winner_id = env.unwrapped.rules.game_result
        if winner_id is not None and winner_id != -1: # 如果不是平局
            winner_agent = player_mapping[winner_id]
            if winner_agent is agent0:
                stats["wins_agent0"] += 1
                stats["total_score_agent0"] += env.unwrapped.rules.yaku_points[0]
            else:
                stats["wins_agent1"] += 1
                stats["total_score_agent0"] -= env.unwrapped.rules.yaku_points[1]
        else:
            stats["draws"] += 1

    env.close()
    return stats

# 辅助函数，用于创建智能体
def create_agent(agent_type, model_path, seed):
    """根据类型和路径创建智能体实例。"""
    if agent_type == "random":
        return RandomAgent(seed=seed)
    elif agent_type == "ppo":
        if not model_path:
            raise ValueError("Must provide a model path for PPO agent.")
        return PPOAgent(model_path=model_path)
    elif agent_type == "rule":
        return RuleAgent()
    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'")

def main():
    """主执行函数"""
    print("Setting up agents for evaluation...")
    agent0 = create_agent(AGENT_0_TYPE, AGENT_0_PATH, SEED)
    agent1 = create_agent(AGENT_1_TYPE, AGENT_1_PATH, SEED)

    results = evaluate_duel(agent0, agent1, num_games=NUM_GAMES)
    
    total_games = results['wins_agent0'] + results['wins_agent1'] + results['draws']
    win_rate_agent0 = (results["wins_agent0"] / total_games) * 100 if total_games > 0 else 0
    win_rate_agent1 = (results["wins_agent1"] / total_games) * 100 if total_games > 0 else 0
    draw_rate = (results["draws"] / total_games) * 100 if total_games > 0 else 0
    avg_net_score = results["total_score_agent0"] / total_games if total_games > 0 else 0

    print("\n" + "="*40)
    print("       >>> Final Fair Evaluation Results <<<")
    print("="*40)
    print(f"Agent 0: {agent0.__class__.__name__} ({AGENT_0_TYPE.upper()})")
    if AGENT_0_TYPE == 'ppo': print(f"  - Model: {AGENT_0_PATH}")
    print(f"Agent 1: {agent1.__class__.__name__} ({AGENT_1_TYPE.upper()})")
    if AGENT_1_TYPE == 'ppo': print(f"  - Model: {AGENT_1_PATH}")
    print(f"Total Games Played: {total_games}")
    print("-" * 40)
    print(f"Agent 0 Wins / Agent 1 Wins / Draws: {results['wins_agent0']} / {results['wins_agent1']} / {results['draws']}")
    print(f"Agent 0 Win Rate: {win_rate_agent0:.2f}%")
    print(f"Agent 1 Win Rate: {win_rate_agent1:.2f}%")
    print(f"Draw Rate: {draw_rate:.2f}%")
    print(f"Agent 0 Average Net Score: {avg_net_score:.2f}")
    print("="*40)

if __name__ == '__main__':
    main()