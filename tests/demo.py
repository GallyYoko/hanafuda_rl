import time
import numpy as np

# 确保你的项目已经以可编辑模式安装 (pip install -e .)
from hanafuda_rl.envs.hanafuda_env import HanafudaEnv

def run_demo(env, agent1, agent2, num_episodes=1, render_wait_time=1.0):
    """
    运行一个可视化的游戏对局。

    Args:
        env (gym.Env): 花札游戏环境。
        agent1 (Agent): 玩家0 (我方) 的智能体。
        agent2 (Agent): 玩家1 (对手) 的智能体。
        num_episodes (int): 运行的游戏局数。
        render_wait_time (float): 每一步渲染后等待的秒数，方便观察。
    """
    for i in range(num_episodes):
        print(f"\n--- 开始第 {i+1} 局游戏 ---")
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # 渲染当前游戏状态
            env.render()
            
            # 根据当前玩家决定使用哪个智能体
            current_player = env.rules.current_player
            
            if current_player == 0: # 我方回合
                print(">>> 我方 (Agent) 回合")
                action_mask = info['action_mask']
                action = agent1.select_action(obs, action_mask)
                print(f"我方选择动作: {action}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    print(f"游戏结束！我方得分: {reward}")
            
            else: # 对手回合 (在我们的环境中，对手逻辑封装在 env.step 里)
                  # 为了演示，我们模拟一个“空”动作，触发环境内部的对手逻辑
                print(">>> 对手 (Opponent) 回合")
                # 在我们的架构中，对手的回合是在我方 step() 函数内部处理的。
                # 这里我们假设 agent1 采取一个动作后，环境会自动处理对手回合。
                # 所以这个 else 分支实际上不会被单独进入。
                # 如果未来将对手逻辑解耦，则会使用 agent2。
                pass # 在当前架构下，此分支留空

            # 等待一段时间，方便人类玩家观察
            time.sleep(render_wait_time)

        # 游戏结束后最后渲染一次最终局面
        print("\n--- 最终局面 ---")
        env.render()
        print(f"最终结果: {env.rules.game_result}")
        print(f"最终得分: 我方 {env.rules.yaku_points[0]}, 对手 {env.rules.yaku_points[1]}")
        print("="*40)


class RandomAgent:
    """一个简单的智能体，从合法的动作中随机选择一个。"""
    def __init__(self, action_space):
        self.action_space = action_space
        # 创建一个随机数生成器
        self.rng = np.random.default_rng()

    def select_action(self, observation, action_mask):
        """
        根据动作掩码选择一个合法的随机动作。

        Args:
            observation (dict): 当前的观测，这里未使用。
            action_mask (np.ndarray): 布尔数组，标记了哪些动作是合法的。

        Returns:
            int: 选择的合法动作。
        """
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            # 理论上，在非终止状态下不应发生，但作为保护
            print("警告：没有找到合法动作！")
            return 0 # 返回一个默认动作
        return self.rng.choice(legal_actions)


if __name__ == "__main__":
    # 1. 创建环境
    # render_mode='ansi' 是我们在 env 中实现的渲染模式
    env = HanafudaEnv(render_mode='ansi')

    # 2. 创建两个智能体
    # 注意：在当前的环境实现中，对手逻辑是硬编码在 _opp_turn() 里的。
    # 所以 agent2 实际上不会被用到，但我们为未来的扩展性保留它。
    agent_player0 = RandomAgent(env.action_space)
    opponent_player1 = RandomAgent(env.action_space) # 占位符

    # 3. 运行可视化 Demo
    # 设置 render_wait_time=1.5 意味着每步后暂停1.5秒，给你足够的时间看清盘面
    run_demo(env, agent_player0, opponent_player1, num_episodes=1, render_wait_time=0)