import os
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

# 导入您的花札环境
from hanafuda_rl.envs.hanafuda_env import HanafudaEnv

# --- 1. 定义一些常量 ---
# 日志和模型保存的根目录
LOG_DIR = "results/logs"
MODEL_DIR = "results/models"
# 为这次训练创建一个唯一的文件夹名
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# 总训练步数
TOTAL_TIMESTEPS = 1_000_000 # 先用一个较小的步数测试流程，例如 10万步

def train_agent():
    """
    最简单的训练函数
    """
    # --- 2. 创建并包装环境 ---
    # MaskablePPO 需要一个特殊的包装器来处理 action_mask
    # 我们先创建一个普通的 Gym 环境
    env = HanafudaEnv(render_mode=None) 
    
    # 然后用 ActionMasker 包装它。这个包装器会从 info dict 中提取 "action_mask"
    # 并将其提供给 MaskablePPO
    env = ActionMasker(env, action_mask_fn=lambda env: env.get_action_mask())

    # --- 3. 定义模型 ---
    # MaskablePPO 会自动处理 Dict Observation Space 和 Action Mask
    # 我们使用 MaskableMultiInputActorCriticPolicy，因为它天生支持 Dict 观测空间
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        verbose=1,  # 打印训练过程中的信息
        tensorboard_log=LOG_DIR, # 指定 TensorBoard 日志目录
        learning_rate=3e-4,     # 学习率，一个常用的默认值
        n_steps=128,           # 每次更新模型前，每个环境要跑多少步
        batch_size=64,          # mini-batch 的大小
        gamma=0.99              # 折扣因子
    )

    # --- 4. 开始训练 ---
    print("="*20)
    print("Start training MaskablePPO model...")
    print(f"Total Steps: {TOTAL_TIMESTEPS}")
    print(f"Log file will be save in: {LOG_DIR}/{TIMESTAMP}")
    print("="*20)
    
    # model.learn() 会自动处理训练循环
    # tb_log_name 参数会为这次运行在 tensorboard_log 目录下创建一个子文件夹
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=f"MaskablePPO_{TIMESTAMP}"
    )

    # --- 5. 保存模型 ---
    # 创建模型保存目录（如果不存在）
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"hanafuda_ppo_{TOTAL_TIMESTEPS}.zip")
    model.save(model_path)

    print("="*20)
    print("Training completed!")
    print(f"Model saved in: {model_path}")
    print("="*20)

    # --- 6. (可选) 关闭环境 ---
    env.close()


if __name__ == '__main__':
    # 确保文件夹存在
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_agent()