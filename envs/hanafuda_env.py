import gymnasium as gym
import numpy as np
from .rules import HanafudaRules

class HanafudaEnv(gym.Env):
    """花札Gymnasium环境"""
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rules = HanafudaRules()
        self.render_mode = render_mode

        # 观察空间定义
        self.observation_space = gym.spaces.Dict({
            # 48张牌的各状态（手牌/收集牌/场牌/已见牌）
            'own_hand': gym.spaces.MultiBinary(48),
            'own_collected': gym.spaces.MultiBinary(48),
            'opp_collected': gym.spaces.MultiBinary(48),
            'table': gym.spaces.MultiBinary(48),
            # 其他标量信息
            'deck_remaining': gym.spaces.Box(0, 32, dtype=np.int32),
            'current_score_self': gym.spaces.Box(-100, 100, dtype=np.int32),
            'current_score_opp': gym.spaces.Box(-100, 100, dtype=np.int32),
            'koikoi_flag_self': gym.spaces.Discrete(2),
            'koikoi_flag_opp': gym.spaces.Discrete(2),
        })

        # 动作空间定义（最大动作数=手牌数*场牌选择）
        self.action_space = gym.spaces.Discrete(8*8 + 2)  # +2用于结束回合/叫牌

    def reset(self, seed=None, options=None):
        """重置游戏状态"""
        super().reset(seed=seed)
        self.rules.reset()
        self.current_player = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """执行动作"""
        # 解析动作
        if action < 8*8:  # 普通出牌动作
            card_idx = action // 8
            table_choice = action % 8
            self.rules.play_card(
                player=self.current_player,
                card_id=self.rules.player_hands[self.current_player][card_idx].card_id,
                table_choice=table_choice,
                drawn_choice=0
            )
        else:  # 特殊动作（结束/叫牌）
            pass  # 待实现

        # 切换玩家
        self.current_player = 1 - self.current_player

        # 检查游戏结束条件
        terminated = self._check_termination()
        reward = self._calculate_reward()
        
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        """渲染当前状态"""
        if self.render_mode == 'human':
            print(f"当前玩家: {self.current_player}")
            print(f"得分: 玩家0={self.rules.yaku_points[0]}, 玩家1={self.rules.yaku_points[1]}")
        elif self.render_mode == 'ansi':
            return str(self.rules)

    def _get_obs(self):
        """获取当前观察"""
        obs = {
            'own_hand': np.zeros(48, dtype=np.int8),
            'own_collected': np.zeros(48, dtype=np.int8),
            'opp_collected': np.zeros(48, dtype=np.int8),
            'table': np.zeros(48, dtype=np.int8),
            'deck_remaining': len(self.rules.draw_pile),
            'current_score_self': self.rules.yaku_points[self.current_player],
            'current_score_opp': self.rules.yaku_points[1 - self.current_player],
            'koikoi_flag_self': 0,  # 待实现
            'koikoi_flag_opp': 0,   # 待实现
        }
        # 填充牌状态（简化示例）
        for card in self.rules.player_hands[self.current_player]:
            obs['own_hand'][card.card_id] = 1
        # 其他牌状态填充...
        return obs

    def _get_info(self):
        """获取调试信息"""
        return {
            'valid_actions': self._get_valid_actions(),
            'yaku_formed': self.rules.check_yaku(self.current_player),
        }

    def _get_valid_actions(self):
        """生成合法动作掩码"""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        # 待实现具体逻辑
        return mask

    def _check_termination(self):
        """检查游戏结束条件"""
        return False  # 待实现

    def _calculate_reward(self):
        """计算即时奖励"""
        return 0  # 待实现