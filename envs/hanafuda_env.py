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
            "deck_remaining": self._deck_remaining,
            "current_scores": self._current_scores,
            "koikoi_flags": self._koikoi_flags,
            "my_yaku_progress": self._my_yaku_progress,
            "opp_yaku_progress": self._opp_yaku_progress,
            "turn_phase": self._turn_phase,
        }
    
    def _get_info(self):
        """
        输出得分信息。
        """
        return {
            "action_mask": self.get_action_mask()
        }

    def reset(self, seed = None):
        """
        重置游戏状态，返回初始 observation 和 info。
        """
        super().reset(seed=seed)
        self.rules.reset(np_random=self.np_random)

        # 从规则引擎获取初始手牌和场牌
        self._hand = np.zeros(48, dtype=np.int8)
        self._table = np.zeros(48, dtype=np.int8)
        hand = self.rules.player_hands[0]
        table = self.rules.table_cards

        # 更新手牌状态
        for card in hand:
            self._hand[card.card_id] = 1

        # 更新场牌状态
        for card in table:
            self._table[card.card_id] = 1

        # 更新收集区信息
        self._my_collected = np.zeros(48, dtype=np.int8)
        self._opp_collected = np.zeros(48, dtype=np.int8)

        # 更新其他信息
        self._drawn_card = 48
        self._deck_remaining = np.array([1.], dtype=np.float32)
        self._current_scores = np.zeros(2, dtype=np.float32)
        self._koikoi_flags = np.zeros(2, dtype=np.int8)
        self._my_yaku_progress = np.zeros(11, dtype=np.float32)
        self._opp_yaku_progress = np.zeros(11, dtype=np.float32)
        self._turn_phase = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        执行动作，返回新的状态、奖励、是否终止、是否截断和额外信息。
        """
        mask = self.get_action_mask()
        if not mask[action]:
            # 如果出现非法动作，不报错，而是惩罚智能体并保持状态不变
            # 这使得环境能兼容 check_env 和不支持掩码的算法
            reward = -1.0  # 负奖励
            terminated = False # 或者 True，取决于你希望智能体如何学习
            truncated = False
            
            # 返回当前状态，让智能体知道它的动作没有产生效果
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        terminated = False # 游戏继续

        if self._turn_phase == 0:  # 出牌阶段

            if not self.rules.player_hands[0]:
                terminated = True  # 游戏结束

                reward = 0.
                
            else:
                # 解析动作
                play_card_idx, match_choice = self._action[action]    
                # 获取玩家手牌中的目标牌
                card_to_play = self.rules.player_hands[0][play_card_idx]            
                # 调用规则引擎的出牌逻辑
                self.rules.play_card(card_to_play.card_id, match_choice)
                # 游戏阶段更新
                if self.rules.turn_phase == 2:
                    self._turn_phase = 2  # 出牌后叫牌阶段
                else:
                    self._turn_phase = 1  # 抽牌阶段
                    # 抽牌动作
                    self.rules.draw_card()
                    # 抽牌更新
                    self._drawn_card = self.rules.drawn_card.card_id

                # 计算奖励
                reward = 0.
        
        elif self._turn_phase == 1:  # 抽牌阶段
            # 解析动作
            draw_choice = self._action[action]
            # 调用规则引擎的配对逻辑
            self.rules.judge_draw_card(draw_choice)
            # 游戏阶段更新
            if self.rules.turn_phase == 3:
                self._turn_phase = 3  # 抽牌后叫牌阶段
            else:
                self.rules.switch_player() # 切换玩家
                self._opp_turn() # 对手回合
                self.rules.switch_player() # 切换玩家
                self._turn_phase = 0  # 出牌阶段
            
            # 计算奖励
            reward = 0.
        
        elif self._turn_phase == 2:  # 出牌后叫牌阶段
            # 解析动作
            koikoi = self._action[action]
            # 更新叫牌标志
            self._koikoi_flags[0] = koikoi
            if koikoi == 1:  # 选择继续
                self.rules.judge_koikoi(True)
                # 游戏阶段更新
                self._turn_phase = 1  # 抽牌阶段
                # 抽牌动作
                self.rules.draw_card()
                # 抽牌更新
                self._drawn_card = self.rules.drawn_card.card_id
                
                # 计算奖励
                reward = 0.
            else:  # 选择不继续
                self.rules.judge_koikoi(False)
                # 游戏结束
                terminated = True

                # 奖励为当前役分
                reward = self._current_scores[0]
        
        else:  # 抽牌后叫牌阶段
            # 解析动作
            koikoi = self._action[action]
            # 更新叫牌标志
            self._koikoi_flags[0] = koikoi
            if koikoi == 1:  # 选择继续
                self.rules.judge_koikoi(True)
                # 游戏阶段更新
                self.rules.switch_player() # 切换玩家
                self._opp_turn()  # 对手回合
                self.rules.switch_player() # 切换玩家
                self._turn_phase = 0  # 出牌阶段

                # 计算奖励
                reward = 0.
            else:  # 选择不继续
                self.rules.judge_koikoi(False)
                # 游戏结束
                terminated = True

                # 奖励为当前役分
                reward = self.rules.yaku_points[0]

        # 检查对手是否结束
        if self.rules.game_over and not terminated:
            terminated = True
            # 如果对手赢了
            if self.rules.game_result == 'player1_win':
            # 惩罚为对手分数
                reward = -self.rules.yaku_points[1]
            # 如果是平局
            else:
                reward = 0.
        
        # 更新环境状态
        self._update_obs()

        # 返回新状态
        observation = self._get_obs()
        info = self._get_info()
        truncated = False
        return observation, reward, terminated, truncated, info

    def get_action_mask(self):
        """
        根据当前游戏阶段和局面，生成合法的动作掩码。
        """
        mask = np.zeros(38, dtype=bool)  # 默认所有动作都非法

        if self._turn_phase == 0:  # 0: 出牌阶段
            # 遍历8个手牌槽位
            hand = self.rules.player_hands[0]
            for i, hand_card in enumerate(hand):
                # 查找场上同月牌
                matching_table_cards = [
                    card for card in self.rules.table_cards
                    if card.month == hand_card.month
                ]
                
                if not matching_table_cards:
                    # 如果没有同月牌，只有“不配对”（选项3）是合法的
                    mask[i * 4 + 3] = True
                else:
                    # 如果有同月牌，可以选择配对其中任意一张
                    for j in range(len(matching_table_cards)):
                        mask[i * 4 + j] = True
        
        elif self._turn_phase == 1:  # 1: 抽牌配对阶段
            # 获取抽出的牌
            drawn_card = self.rules.drawn_card
            # 查找场上同月牌
            matching_table_cards = [
                card for card in self.rules.table_cards
                if card.month == drawn_card.month
            ]

            if not matching_table_cards:
                # 如果没有同月牌，只有“不配对”（选项3，对应动作ID 35）是合法的
                mask[32 + 3] = True
            else:
                # 如果有同月牌，可以选择配对其中任意一张
                for j in range(len(matching_table_cards)):
                    mask[32 + j] = True

        elif self._turn_phase == 2 or self._turn_phase == 3:  # 2 & 3: 叫牌决策阶段
            # 动作36（叫牌）和37（结束）都是合法的
            mask[36] = True
            mask[37] = True
            
        return mask

    def _update_obs(self):
        """
        更新状态。
        """
        hand = self.rules.player_hands[0]
        table = self.rules.table_cards
        collected_cards = self.rules.collected_cards

        # 清空牌组状态
        self._hand.fill(0)
        self._table.fill(0)
        self._my_collected.fill(0)
        self._opp_collected.fill(0)

        # 更新手牌状态
        for card in hand:
            self._hand[card.card_id] = 1

        # 更新场牌状态
        for card in table:
            self._table[card.card_id] = 1

        # 更新收集牌状态
        for card in collected_cards[0]:
            self._my_collected[card.card_id] = 1
        for card in collected_cards[1]:
            self._opp_collected[card.card_id] = 1

        # 更新抽牌
        if self._turn_phase == 1:
            self._drawn_card = self.rules.drawn_card.card_id
        else:
            self._drawn_card = 48

        # 更新山牌剩余数
        self._deck_remaining[0] = len(self.rules.draw_pile) / 24

        # 更新当前役分
        self._current_scores[0] = self.rules.yaku_points[0]
        self._current_scores[1] = self.rules.yaku_points[1]

        # 更新叫牌情况
        self._koikoi_flags[0] = self.rules.koikoi_flags[0]
        self._koikoi_flags[1] = self.rules.koikoi_flags[1]

        # 更新役进度
        self._my_yaku_progress = self.rules.yaku_progress[0]
        self._opp_yaku_progress = self.rules.yaku_progress[1]

    def _opp_turn(self):
        """
        对手回合：调用引擎模拟对手的动作（随机打出手牌）。
        """
         # 确保当前玩家是对手 (player 1)
        if self.rules.current_player != 1:
            raise ValueError(f"Illegal player: {self.rules.current_player}")
        
        # 对手出牌阶段
        opp_hand = self.rules.player_hands[1]
        if not opp_hand:
            self.rules.game_over = True
            return
        
        # 随机选择一张手牌打出
        card_to_play = self.np_random.choice(opp_hand)

        # 查找场上所有合法的配对选项
        table_matches = [card for card in self.rules.table_cards if card.month == card_to_play.month]

        if not table_matches:
        # 没有可配对的牌，只能选择“不配对”（选项3）
            match_choice = 3
        else:
            # 从合法的配对选项中随机选择一个
            match_choice = self.np_random.integers(0, len(table_matches))

        # 执行出牌动作，这会更新牌局并可能改变 turn_phase
        self.rules.play_card(card_to_play.card_id, match_choice)

        if self.rules.turn_phase == 2:  # 轮到对手叫牌
            # 随机决定是否叫牌 (50%概率)
            if self.np_random.choice([True, False]):
                # 决定叫牌 (Koi-Koi)
                self.rules.judge_koikoi(True)
                self.rules.turn_phase = 0 # 状态重置，准备抽牌
            else:
                # 决定结束游戏
                self.rules.judge_koikoi(False)
                return # 对手回合结束，游戏结束
            
        self.rules.draw_card() # 从牌堆抽牌
        drawn_card = self.rules.drawn_card

        # 查找抽出的牌在场上的配对选项
        draw_matches = [card for card in self.rules.table_cards if card.month == drawn_card.month]

        if not draw_matches:
            draw_choice = 3 # 不配对
        else:
            draw_choice = self.np_random.integers(0, len(draw_matches))

        # 执行抽牌配对，这会再次更新牌局并可能改变 turn_phase
        self.rules.judge_draw_card(draw_choice)

        if self.rules.turn_phase == 3: # 轮到对手叫牌
            # 再次随机决定是否叫牌
            if self.np_random.choice([True, False]):
                self.rules.judge_koikoi(True)
                # 对手叫牌后，其回合结束
            else:
                self.rules.judge_koikoi(False)
                return # 对手回合结束，游戏结束
            
        # 检查对手手牌是否在此回合中打完
        if not self.rules.player_hands[1]:
            self.rules.game_over = True
            return

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