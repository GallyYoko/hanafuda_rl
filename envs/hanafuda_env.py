import gymnasium as gym
import numpy as np
from .rules import HanafudaRules


class HanafudaEnv(gym.Env):
    """
    花札游戏环境，基于 Gymnasium 接口实现。
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode
        self.rules = HanafudaRules()

        # 定义观测量
        self._hand = np.zeros(48, dtype=np.int8)  # 我方手牌
        self._table = np.zeros(48, dtype=np.int8)  # 场牌
        self._my_collected = np.zeros(48, dtype=np.int8)  # 我方收集牌
        self._opp_collected = np.zeros(48, dtype=np.int8)  # 对手收集牌
        self._drawn_card = 48  # 抽中的牌
        self._current_scores = np.zeros(2, dtype=np.float32)  # 当前役分
        self._turn_phase = 0  # 游戏阶段

        # 定义状态集（observations）
        self.observation_space = gym.spaces.Dict(
            {
                # 牌面信息（multi-hot）
                "hand": gym.spaces.MultiBinary(48),  # 我方手牌
                "table": gym.spaces.MultiBinary(48),  # 场牌
                "my_collected": gym.spaces.MultiBinary(48),  # 我方收集牌
                "opp_collected": gym.spaces.MultiBinary(48),  # 对手收集牌
                "drawn_card": gym.spaces.Discrete(49),  # 抽中的牌

                # 局面数值特征
                "deck_remaining": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 山牌剩余数
                "current_scores": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # 当前役分（归一化）
                "koikoi_flags": gym.spaces.MultiBinary(2),  # 是否叫牌（双方）

                # 役进度特征
                "yaku_progress": gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32),

                # 游戏阶段
                "turn_phase": gym.spaces.Discrete(3),  # 游戏阶段阶段（出牌、抽牌、叫牌）
            }
        )

        # 定义动作集
        self.action_space = gym.spaces.Discrete(8*4+4+2)  # 手牌选择 * 场牌配对 + 抽牌配对 + 叫牌
    
    def _get_obs(self):
        """
        输出当前状态。
        """
        return {
            "hand": self._hand, 
            "table": self._table,
            "my_collected": self._my_collected,
            "opp_collected": self._opp_collected,
            "drawn_card": self._drawn_card,
            "turn_phase": self._turn_phase,
        }
    
    def _get_info(self):
        """
        输出得分信息。
        """
        return {"current_scores": self._current_scores}

    def reset(self, seed = None):
        """
        重置游戏状态，返回初始 observation 和 info。
        """
        super().reset(seed=seed)
        self.rules.reset(seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        执行动作，返回新的状态、奖励、是否终止、是否截断和额外信息。
        """
        # TODO: 实现动作逻辑
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        渲染当前游戏状态。
        """
        if self.render_mode == "human":
            # TODO: 实现可视化
            pass
        elif self.render_mode == "ansi":
            # TODO: 实现 ASCII 渲染
            pass

    def close(self):
        """
        清理环境资源。
        """
        pass