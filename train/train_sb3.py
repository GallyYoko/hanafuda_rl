import os
from datetime import datetime
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# 导入花札环境和随机智能体
from hanafuda_rl.envs.hanafuda_env import HanafudaEnv
from hanafuda_rl.agents.random_agent import RandomAgent
from hanafuda_rl.agents.sb3_agent import PPOAgent

# --- 1. 定义一个包装器，用于处理“RL vs 对手”的逻辑 ---
class SelfPlayEnvWrapper(gym.Wrapper):
    """
    一个包装器，让一个RL智能体可以和另一个固定策略的智能体对战。
    这个包装器将二人游戏转换为对于RL智能体来说的单人游戏。
    """
    def __init__(self, env, opponent_agent):
        super().__init__(env)
        self.opponent_agent = opponent_agent
        self.rl_player_id = 0  # 假定RL智能体总是玩家0

    def get_action_mask(self):
        """
        将底层环境的 get_action_mask 方法暴露出来。
        MaskablePPO 会检查这个方法是否存在来确认环境是否支持掩码。
        """
        return self.env.unwrapped.get_action_mask()

    def reset(self, **kwargs):
        """
        重置环境，并确保如果对手先手，则让其完成回合。
        """
        obs, info = self.env.reset(**kwargs)
        # 如果开局是对手先手
        if self.env.unwrapped.current_player != self.rl_player_id:
            # 让对手一直玩，直到轮到我们
            obs, _, _, _, info = self._opponent_play_until_our_turn(obs, info)
        return obs, info

    def step(self, action):
        """
        RL智能体执行一步，然后让对手一直玩，直到再次轮到RL智能体。
        """
        # RL智能体执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 如果游戏没有因RL智能体的动作而结束，并且轮到对手了
        opp_reward = 0.
        if not (terminated or truncated) and self.env.unwrapped.current_player != self.rl_player_id:
            obs, opp_reward, terminated, truncated, info = self._opponent_play_until_our_turn(obs, info)

        if terminated or truncated:
            reward = reward - opp_reward # 用对手的得分修正RL智能体的得分
        
        return obs, reward, terminated, truncated, info

    def _opponent_play_until_our_turn(self, obs, info):
        """
        让对手一直行动，直到再次轮到RL智能体或者游戏结束。
        这个辅助函数返回完整的5元组，供 step 和 reset 方法内部使用。
        """
        last_reward = 0.
        terminated, truncated = False, False

        # 只要当前玩家是对手，并且游戏没有结束
        while self.env.unwrapped.current_player != self.rl_player_id:
            # 从环境中获取最新的掩码给对手
            action_mask = self.env.unwrapped.get_action_mask()
            
            # 使用对手的策略选择动作
            opponent_action = self.opponent_agent.select_action(obs, action_mask)
            
            # 在环境中执行对手的动作
            obs, reward, terminated, truncated, info = self.env.step(opponent_action)
            last_reward = reward

            if terminated or truncated:
                break
        
        # 如果循环从未执行（例如，对手回合开始时游戏就已结束），obs会是None
        # 在这种情况下，我们需要从环境中获取一次当前的观测状态
        if obs is None:
            obs = self.env.unwrapped._get_obs(self.rl_player_id)
        
        return obs, last_reward, terminated, truncated, info

# 定义常量
LOG_DIR = "Hanafuda-Project/hanafuda_rl/results/logs"
MODEL_DIR = "Hanafuda-Project/hanafuda_rl/results/models"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TOTAL_TIMESTEPS = 5_000_000
N_STEPS = 2048
N_ENVS = 10 # 多线程并行
SEED = 99

# 为自我对弈设置参数
SELF_PLAY_ITERATIONS = 5 # 自我对弈的总迭代轮数
STEPS_PER_ITERATION = TOTAL_TIMESTEPS // SELF_PLAY_ITERATIONS # 每轮迭代训练的步数

# 一个辅助函数，用于创建和包装单个环境实例
def make_env_func(rank, seed=99, opponent_model_path=None):
    """
    一个辅助函数，返回一个创建环境的函数。
    这是 SubprocVecEnv 所需的格式。
    """
    def _init():
        env_seed = seed + rank

        # 动态决定对手
        if opponent_model_path is None:
            # 如果没有提供模型路径，则使用随机智能体（用于第一轮训练）
            opponent_seed = seed + rank
            opponent = RandomAgent(seed=opponent_seed)
        else:
            # 如果提供了模型路径，则加载该PPO模型作为对手
            opponent = PPOAgent(model_path=opponent_model_path)

        # 创建环境
        env = HanafudaEnv()
        env.reset(seed=env_seed)
        
        # 按顺序包装
        env = SelfPlayEnvWrapper(env, opponent_agent=opponent)
        env = ActionMasker(env, action_mask_fn=lambda env: env.get_action_mask())
        env = Monitor(env)
        return env
    return _init

def train_agent():
    """主训练函数"""
    
    # 初始化模型和对手路径
    model = None
    opponent_path = None # 第一轮没有对手模型

    for i in range(SELF_PLAY_ITERATIONS):
        print("="*50)
        print(f"Starting Self-Play Iteration {i+1}/{SELF_PLAY_ITERATIONS}")
        print(f"Opponent: {'RandomAgent' if opponent_path is None else opponent_path}")
        print("="*50)

        # 1. 根据当前对手创建并行环境
        vec_env = SubprocVecEnv([make_env_func(rank, SEED, opponent_model_path=opponent_path) for rank in range(N_ENVS)])

        # 2. 创建或更新模型
        if model is None:
            # 如果是第一次迭代，创建一个新模型
            print("Creating a new MaskablePPO model...")
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                vec_env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=3e-4,
                n_steps=N_STEPS,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
            )
        else:
            # 如果不是第一次迭代，更新模型以使用新的环境（新的对手）
            print("Updating model with new environment (new opponent)...")
            model.set_env(vec_env)

        # 3. 训练模型
        # reset_num_timesteps=False 确保日志和总步数在迭代之间是连续的
        model.learn(
            total_timesteps=STEPS_PER_ITERATION,
            tb_log_name=f"MaskablePPO_SelfPlay_{TIMESTAMP}",
            progress_bar=True,
            reset_num_timesteps=False 
        )

        # 4. 保存当前模型，它将成为下一轮的对手
        current_model_path = os.path.join(MODEL_DIR, f"selfplay_models/hanafuda_ppo_iter_{i+1}.zip")
        model.save(current_model_path)
        print(f"Iteration {i+1} model saved to: {current_model_path}")

        # 5. 更新对手路径以备下一轮使用
        opponent_path = current_model_path

        # 6. 关闭当前的环境，释放资源
        vec_env.close()

    print("="*50)
    print("Self-Play training completed!")
    final_model_path = os.path.join(MODEL_DIR, f"hanafuda_ppo_selfplay_{TOTAL_TIMESTEPS}.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print("="*50)

if __name__ == '__main__':
    # 确保文件夹存在
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 开始训练
    train_agent()