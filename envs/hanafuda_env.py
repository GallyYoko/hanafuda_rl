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
        self._deck_remaining = np.array([1.], dtype=np.float32)  # 山牌剩余数
        self._current_scores = np.zeros(2, dtype=np.float32)  # 当前役分
        self._koikoi_flags = np.zeros(2, dtype=np.int8)  # 是否叫牌（双方）
        self._my_yaku_progress = np.zeros(11, dtype=np.float32)  # 我方役进度表
        self._opp_yaku_progress = np.zeros(11, dtype=np.float32)  # 对手役进度表
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
                "my_yaku_progress": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),
                "opp_yaku_progress": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),

                # 游戏阶段
                "turn_phase": gym.spaces.Discrete(4),  # 游戏阶段阶段（出牌、抽牌、出牌后叫牌、抽牌后叫牌）
            }
        )

        # 定义动作
        self._action = {}
        # 出牌动作
        for play_card in range(8):
            for match_card in range(4):
                self._action[play_card*4 + match_card] = [play_card, match_card]
        # 抽牌动作
        for draw_match_card in range(4):
            self._action[32 + draw_match_card] = draw_match_card
        # 叫牌动作
        for koikoi in range(2):
            self._action[36 + koikoi] = koikoi # (36, 37) 分别为不叫牌和叫牌

        # 定义动作集
        self.action_space = gym.spaces.Discrete(8*4+4+2)  # 手牌选择 * 场牌配对 + 抽牌配对 + 叫牌

        # 跟踪当前玩家
        self.current_player = 0 
    
    def _get_obs(self, player_id):
        """
        更新并输出当前状态（指定玩家）。
        """
        opp_id = 1 - player_id
        # 更新阶段
        self._turn_phase = self.rules.turn_phase

        # 从规则引擎获取全局信息
        hand = self.rules.player_hands[player_id]
        table = self.rules.table_cards
        my_collected_cards = self.rules.collected_cards[player_id]
        opp_collected_cards = self.rules.collected_cards[opp_id]

        # 清空牌组状态
        self._hand.fill(0)
        self._table.fill(0)
        self._my_collected.fill(0)
        self._opp_collected.fill(0)

        # 更新牌面信息
        for card in hand: self._hand[card.card_id] = 1
        for card in table: self._table[card.card_id] = 1
        for card in my_collected_cards: self._my_collected[card.card_id] = 1
        for card in opp_collected_cards: self._opp_collected[card.card_id] = 1

        # 更新抽牌
        if self._turn_phase == 1:
            self._drawn_card = self.rules.drawn_card.card_id
        else:
            self._drawn_card = 48

        # 更新山牌剩余数
        self._deck_remaining[0] = len(self.rules.draw_pile) / 24

        # 更新当前役分
        self._current_scores[0] = np.tanh(self.rules.yaku_points[player_id] / 5.0)
        self._current_scores[1] = np.tanh(self.rules.yaku_points[opp_id] / 5.0)

        # 更新叫牌情况
        self._koikoi_flags[0] = self.rules.koikoi_flags[player_id]
        self._koikoi_flags[1] = self.rules.koikoi_flags[opp_id]

        # 更新役进度
        self._my_yaku_progress = self.rules.yaku_progress[player_id]
        self._opp_yaku_progress = self.rules.yaku_progress[opp_id]

        return {
            "hand": self._hand, 
            "table": self._table,
            "my_collected": self._my_collected,
            "opp_collected": self._opp_collected,
            "drawn_card": self._drawn_card,
            "deck_remaining": self._deck_remaining,
            "current_scores": self._current_scores,
            "koikoi_flags": self._koikoi_flags,
            "my_yaku_progress": self._my_yaku_progress,
            "opp_yaku_progress": self._opp_yaku_progress,
            "turn_phase": self._turn_phase,
        }
    
    def _get_info(self, reward = 0):
        """
        输出动作掩码和当前玩家信息。
        """
        if not self.rules.game_over:
            reward_dict = {self.current_player: reward, 1 - self.current_player: 0}
        else:
            reward_dict = {self.current_player: reward, 1 - self.current_player: -reward}
        return {
            "action_mask": self.get_action_mask(),
            "current_player": self.current_player,
            "reward_dict": reward_dict
        }

    def reset(self, seed = None, options=None):
        """
        重置游戏状态，返回初始 observation 和 info。
        """
        super().reset(seed=seed)
        self.rules.reset(np_random=self.np_random)
        self.current_player = self.rules.current_player
        self._turn_phase = 0

        observation = self._get_obs(self.current_player)
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        执行动作，返回新的状态、奖励、是否终止、是否截断和额外信息。
        """
        player_id = self.rules.current_player
        former_points = self.rules.yaku_points[player_id]
        mask = self.get_action_mask()

        terminated = False # 游戏继续
        truncated = False

        if not mask[action]:
            # 如果出现非法动作，不报错，而是惩罚智能体并保持状态不变
            # 这使得环境能兼容 check_env 和不支持掩码的算法
            reward = -1.0  # 负奖励
            terminated = False # 或者 True，取决于你希望智能体如何学习
            truncated = False

        else:
            self.rules.perform_action(action, player_id)
            latter_points = self.rules.yaku_points[player_id]

            terminated = self.rules.game_over # 判断终止
            reward = self._calculate_reward(former_points, latter_points) # 计算奖励

            self.current_player = self.rules.current_player # 更新当前玩家

        # 返回新状态
        observation = self._get_obs(self.current_player)
        info = self._get_info(reward)

        return observation, reward, terminated, truncated, info

    def get_action_mask(self):
        """
        根据当前游戏阶段和局面，生成合法的动作掩码。
        """
        mask = self.rules.get_legal_actions_mask(self.current_player)
            
        return mask

    def _calculate_reward(self, former_points, latter_points):
        """
        计算奖励。
        """
        if not self.rules.game_over:
            if self._turn_phase in [0,1]: # 出牌或抽牌阶段
                return (latter_points - former_points) * 0.1
            else: # 叫牌阶段
                if self.rules.koikoi_flags[self.current_player] == 1: # 智能体叫牌
                    return 0.
                else: # 智能体不叫牌，游戏结束
                    return self.rules.yaku_points[self.current_player]
        else:
            if self.rules.game_result == self.current_player: # 智能体获胜
                return self.rules.yaku_points[self.current_player]
            elif self.rules.game_result == 1 - self.current_player: # 智能体失败
                return - self.rules.yaku_points[1 - self.current_player]
            else:
                return 0.

    def render(self):
        """
        以 ANSI 文本格式渲染当前游戏状态。
        使用紧凑的单行卡牌表示法 [月份-类别]。
        """
        if self.render_mode == "ansi":
            # --- 辅助函数，使用新的紧凑格式打印单行 ---
            def print_cards_compact(cards):
                if not cards:
                    print("(空)")
                    return
                
                # 使用新的紧凑格式
                card_strings = [self._card_to_compact_str(c) for c in cards]
                print(" ".join(card_strings))

            # 清空屏幕
            print("-"*90)
            
            # --- 对手信息 ---
            print("--- 对手 (Player 1) ---")
            print(f"手牌数: {len(self.rules.player_hands[1])}")
            print("收集的牌:")
            print_cards_compact(self.rules.collected_cards[1])
            
            print("\n" + "-"*42 + " 场面 " + "-"*41)
            
            # --- 场牌 ---
            print_cards_compact(self.rules.table_cards)

            # --- 我方信息 ---
            print("\n--- 我方 (Agent, Player 0) ---")
            print("手牌:")
            print_cards_compact(self.rules.player_hands[0])
            print("收集的牌:")
            print_cards_compact(self.rules.collected_cards[0])

            print("\n" + "-"*90)

            # --- 游戏状态 ---
            phase_map = {
                0: "出牌阶段",
                1: "抽牌配对阶段",
                2: "出牌后叫牌决策",
                3: "抽牌后叫牌决策"
            }
            # 获取役种列表
            my_yakus = self.rules.yaku_list[0]
            opp_yakus = self.rules.yaku_list[1]
            
            # 构建包含役种名称的得分字符串
            my_score_str = f"我方得分: {self.rules.yaku_points[0]}"
            if my_yakus:
                my_score_str += f"  (役: {', '.join(my_yakus)})" # 添加役种

            opp_score_str = f"对手得分: {self.rules.yaku_points[1]}"
            if opp_yakus:
                opp_score_str += f"  (役: {', '.join(opp_yakus)})" # 添加役种
                
            print(my_score_str)
            print(opp_score_str)

            # 打印其他信息
            deck_str = f"山牌剩余: {len(self.rules.draw_pile)}"
            phase_str = f"当前阶段: {phase_map.get(self._turn_phase, '未知')}"
            
            if self._turn_phase == 1 and self.rules.drawn_card:
                # 叫牌决策时，也显示一下是什么牌触发的
                drawn_card_str = f" -> 抽到: {self._card_to_compact_str(self.rules.drawn_card)}"
                phase_str += drawn_card_str

            print(f"{my_score_str:<30} {opp_score_str:<30} {deck_str:<25}")
            print(phase_str)
            print("="*90 + "\n")

        elif self.render_mode == "human":
            print("Human render mode is not implemented yet. Use 'ansi'.")
            pass
    
    def _card_to_compact_str(self, card):
        """将 Card 对象转换为一个紧凑的字符串表示，例如 [01-光]"""
        if card is None:
            return ""
        
        month_str = str(card.month).zfill(2)
        # 使用类别首字母来缩写
        category_map = {
            "光": "光",
            "种": "种",
            "短册": "短",
            "佳士": "佳"
        }
        cat_str = category_map.get(card.category, "?")
        
        return f"[{month_str}-{cat_str}]"

    def close(self):
        """
        清理环境资源。
        """
        pass