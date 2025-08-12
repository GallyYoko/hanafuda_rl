import gymnasium as gym
from tqdm import tqdm

# 确保项目已以可编辑模式安装 (pip install -e .)
from hanafuda_rl.envs.hanafuda_env import HanafudaEnv
from hanafuda_rl.agents.random_agent import RandomAgent
from hanafuda_rl.agents.sb3_agent import PPOAgent

# 'random' -> 评估随机智能体
# 'ppo'    -> 评估已训练的 PPO 模型
AGENT_TYPE_TO_EVALUATE = 'ppo' 

# 如果 AGENT_TYPE_TO_EVALUATE 设置为 'ppo', 请在这里填写你的模型路径
# 例如: "results/models/hanafuda_ppo_1.zip"
PPO_MODEL_PATH = "Hanafuda-Project/hanafuda_rl/results/models/hanafuda_ppo_2.zip"

# 设置评估的游戏总局数
NUM_GAMES_TO_EVALUATE = 1000

# 设置种子
SEED = 99

def evaluate(agent, num_games=1000):
    """评估一个智能体对战环境内置随机对手的性能。"""
    print(f"Evaluating {agent.__class__.__name__} vs. the built-in random agent for {num_games} games...")
    
    env = HanafudaEnv()
    
    wins, losses, draws = 0, 0, 0
    total_net_score = 0
    
    for _ in tqdm(range(num_games), desc="Evaluation Progress"):
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            action_mask = info['action_mask']
            action = agent.select_action(obs, action_mask)
            obs, _, terminated, _, info = env.step(action)

        if env.rules.game_result == 'player0_win':
            wins += 1
        elif env.rules.game_result == 'player1_win':
            losses += 1
        else:
            draws += 1
            
        net_score = env.rules.yaku_points[0] - env.rules.yaku_points[1]
        total_net_score += net_score
            
    env.close()

    win_rate = (wins / num_games) * 100
    loss_rate = (losses / num_games) * 100
    draw_rate = (draws / num_games) * 100
    avg_net_score = total_net_score / num_games

    results = {
        "win_rate": win_rate, "loss_rate": loss_rate, "draw_rate": draw_rate,
        "avg_net_score": avg_net_score, "total_games": num_games,
        "wins": wins, "losses": losses, "draws": draws,
    }
    
    return results

def main():
    """主执行函数"""
    
    # 根据配置创建智能体
    if AGENT_TYPE_TO_EVALUATE == "random":
        agent_to_evaluate = RandomAgent(action_space=gym.spaces.Discrete(38), seed=SEED)
    elif AGENT_TYPE_TO_EVALUATE == "ppo":
        if not PPO_MODEL_PATH:
            raise ValueError("Must provide PPO model path in PPO_MODEL_PATH for evaluating PPO agent.")
        agent_to_evaluate = PPOAgent(model_path=PPO_MODEL_PATH)
    else:
        raise ValueError(f"Unkown agent type: '{AGENT_TYPE_TO_EVALUATE}'")

    # 运行评估
    evaluation_results = evaluate(agent=agent_to_evaluate, num_games=NUM_GAMES_TO_EVALUATE)
    
    # 打印格式化的结果
    print("\n" + "="*30)
    print("       >>> Final Evaluation <<<")
    print("="*30)
    print(f"Agent Type: {AGENT_TYPE_TO_EVALUATE.upper()}")
    if AGENT_TYPE_TO_EVALUATE == "ppo":
        print(f"Model Path: {PPO_MODEL_PATH}")
    print(f"Total Games: {evaluation_results['total_games']}")
    print("-" * 30)
    print(f"Wins / Losses / Draws: {evaluation_results['wins']} / {evaluation_results['losses']} / {evaluation_results['draws']}")
    print(f"Win Rate: {evaluation_results['win_rate']:.2f}%")
    print(f"Loss Rate: {evaluation_results['loss_rate']:.2f}%")
    print(f"Draw Rate: {evaluation_results['draw_rate']:.2f}%")
    print(f"Average Net Score: {evaluation_results['avg_net_score']:.2f}")
    print("="*30)

if __name__ == '__main__':
    main()