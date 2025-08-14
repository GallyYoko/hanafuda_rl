import numpy as np

class RandomAgent:
    def __init__(self, seed = None):
        """
        一个简单的随机智能体，它从合法的动作掩码中随机选择一个动作。
        """
        self.np_random = np.random.default_rng(seed)

    def select_action(self, observation, action_mask):
        """
        根据动作掩码选择一个合法的随机动作。
        """
        # 找到所有合法动作的索引
        legal_actions = np.where(action_mask)[0]
        
        # 如果没有合法动作（理论上游戏结束前不应发生），返回一个默认动作
        if len(legal_actions) == 0:
            # 这是一个警告信号，说明可能存在问题
            print("Warning: No legal actions found in the action mask")
            return 0 
            
        # 从合法动作中随机选择一个
        return self.np_random.choice(legal_actions)