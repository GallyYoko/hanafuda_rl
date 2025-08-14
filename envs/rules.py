import numpy as np

class Card:
    """
    花札牌类，表示一张花札牌。
    """

    def __init__(self, card_id, month, category, points, card_name):
        self.card_id = card_id  # 牌的全局唯一ID（0-47）
        self.month = month      # 月份（1-12，对应1-12月）
        self.category = category  # 牌的类型："光"、"种"、"短册"、"佳士"
        self.points = points    # 牌的基础分数
        self.card_name = card_name  # 牌的名称

    def __repr__(self):
        return f"Card(id={self.card_id}, month={self.month}, category='{self.category}', points={self.points}, name='{self.card_name}')"


class Deck:
    """
    花札牌组类，管理牌的初始化和发牌逻辑。
    """

    def __init__(self):
        self.cards = self._initialize_deck()

    def _initialize_deck(self):
        """
        初始化花札牌组，共48张牌。
        """
        cards = []
        card_id = 0

        # 月份与牌的定义（1-12月）
        months = [
            ("松", ["光", "短册", "佳士", "佳士"]),
            ("梅", ["种", "短册", "佳士", "佳士"]),
            ("樱", ["光", "短册", "佳士", "佳士"]),
            ("藤", ["种", "短册", "佳士", "佳士"]),
            ("菖蒲", ["种", "短册", "佳士", "佳士"]),
            ("牡丹", ["种", "短册", "佳士", "佳士"]),
            ("萩", ["种", "短册", "佳士", "佳士"]),
            ("芒", ["光", "种", "佳士", "佳士"]),
            ("菊", ["种", "短册", "佳士", "佳士"]),
            ("枫", ["种", "短册", "佳士", "佳士"]),
            ("柳", ["光", "种", "短册", "佳士"]),
            ("桐", ["光", "佳士", "佳士", "佳士"]),
        ]

        # 为每个月创建4张牌
        for month_idx, (month_name, categories) in enumerate(months, start=1):
            for category in categories:
                # 根据牌类型分配基础分数
                if category == "光":
                    points = 20
                elif category == "种":
                    points = 10
                elif category == "短册":
                    points = 5
                else:
                    points = 1

                # 根据月份和类别生成牌的名称
                if month_idx == 1:  # 一月 - 松
                    if category == "光":
                        card_name = "松上鹤"
                    elif category == "短册":
                        card_name = "松上赤短"
                    else:
                        card_name = "松"
                elif month_idx == 2:  # 二月 - 梅
                    if category == "种":
                        card_name = "梅上莺"
                    elif category == "短册":
                        card_name = "梅上赤短"
                    else:
                        card_name = "梅"
                elif month_idx == 3:  # 三月 - 樱
                    if category == "光":
                        card_name = "樱上幕"
                    elif category == "短册":
                        card_name = "樱上赤短"
                    else:
                        card_name = "樱"
                elif month_idx == 4:  # 四月 - 藤
                    if category == "种":
                        card_name = "藤上杜鹃"
                    elif category == "短册":
                        card_name = "藤上短册"
                    else:
                        card_name = "藤"
                elif month_idx == 5:  # 五月 - 菖蒲
                    if category == "种":
                        card_name = "溪间八桥"
                    elif category == "短册":
                        card_name = "溪上短册"
                    else:
                        card_name = "溪"
                elif month_idx == 6:  # 六月 - 牡丹
                    if category == "种":
                        card_name = "牡丹上蝶"
                    elif category == "短册":
                        card_name = "牡丹青短"
                    else:
                        card_name = "牡丹"
                elif month_idx == 7:  # 七月 - 萩
                    if category == "种":
                        card_name = "萩间野猪"
                    elif category == "短册":
                        card_name = "萩上短册"
                    else:
                        card_name = "萩"
                elif month_idx == 8:  # 八月 - 芒
                    if category == "光":
                        card_name = "芒上月"
                    elif category == "种":
                        card_name = "芒上雁"
                    else:
                        card_name = "芒"
                elif month_idx == 9:  # 九月 - 菊
                    if category == "种":
                        card_name = "菊上杯"
                    elif category == "短册":
                        card_name = "菊上青短"
                    else:
                        card_name = "菊"
                elif month_idx == 10:  # 十月 - 枫
                    if category == "种":
                        card_name = "枫间鹿"
                    elif category == "短册":
                        card_name = "枫上青短"
                    else:
                        card_name = "枫"
                elif month_idx == 11:  # 十一月 - 柳
                    if category == "光":
                        card_name = "柳间小野道风"
                    elif category == "种":
                        card_name = "柳间燕"
                    elif category == "短册":
                        card_name = "柳上短册"
                    else:
                        card_name = "柳"
                elif month_idx == 12:  # 十二月 - 桐
                    if category == "光":
                        card_name = "桐上凤凰"
                    else:
                        card_name = "桐"
                cards.append(Card(card_id, month_idx, category, points, card_name))
                card_id += 1

        return cards

    def deal(self, np_random = None):
        """
        发牌逻辑：
        - 返回玩家手牌（2×8张）、场牌（8张）、剩余牌堆。
        - 支持随机或确定性发牌（通过seed参数）。
        - 手牌和场牌按card_id排序。
        """
        if np_random is None:
            np_random = np.random.default_rng()
        shuffled_cards = np_random.permutation(self.cards)
        shuffled_cards = list(shuffled_cards)

        # 玩家手牌（每人8张）
        player1_hand = sorted(shuffled_cards[:8], key=lambda card: card.card_id)
        player2_hand = sorted(shuffled_cards[8:16], key=lambda card: card.card_id)

        # 场牌（8张）
        table_cards = sorted(shuffled_cards[16:24], key=lambda card: card.card_id)

        # 剩余牌堆
        draw_pile = shuffled_cards[24:]

        return [player1_hand, player2_hand], table_cards, draw_pile


class HanafudaRules:
    """
    花札规则引擎，处理游戏的核心逻辑。
    """

    def __init__(self):
        self.deck = Deck()
        self.player_hands = None
        self.table_cards = None
        self.draw_pile = None
        self.drawn_card = None
        self.collected_cards = {0: [], 1: []}  # 玩家0和玩家1的收集牌
        self.yaku_points = {0: 0, 1: 0}  # 玩家0和玩家1的役种分数
        self.koikoi_flags = {0: 0, 1: 0}  # 玩家0和玩家1是否叫牌
        self.yaku_list = {0: [], 1: []}  # 玩家0和玩家1的役种列表
        self.yaku_progress = {0: np.zeros(11, dtype=np.float32), 1: np.zeros(11, dtype=np.float32)} # 玩家0和玩家1的役种进度
        self.current_player = 0  # 当前玩家（0或1）
        self.turn_phase = 0  # 当前阶段（0：出牌阶段，1：抽牌阶段，2：出牌后叫牌阶段，3：抽牌后叫牌阶段）
        self.game_over = False  # 游戏是否结束
        self.game_result = None # 游戏是否结束（None：未结束，-1：平局，0：玩家0获胜，1：玩家1获胜）

    def reset(self, np_random = np.random.default_rng()):
        """
        重置游戏状态，返回初始发牌结果。
        同时检查手牌是否符合"手四"或"食付"条件。
        """
        self.player_hands, self.table_cards, self.draw_pile = self.deck.deal(np_random)
        self.collected_cards = {0: [], 1: []}
        self.yaku_points = {0: 0, 1: 0}
        self.koikoi_flags = {0: 0, 1: 0}
        self.yaku_list = {0: [], 1: []}
        self.yaku_progress = {0: np.zeros(11, dtype=np.float32), 1: np.zeros(11, dtype=np.float32)}
        self.drawn_card = None
        self.current_player = np_random.choice([0, 1])
        self.turn_phase = 0
        self.game_over = False
        self.game_result = None
        
        # 检查手四和食付
        for player in range(2):
            # 手四：4张同月份
            month_counts = {}
            for card in self.player_hands[player]:
                month_counts[card.month] = month_counts.get(card.month, 0) + 1
            
            for month, count in month_counts.items():
                if count >= 4:
                    self.yaku_points[player] = 6
                    self.yaku_list[player].append("手四")
                    self.game_over = True
                    self.game_result = player
                    break
            
            # 食付：4对2张同月份
            if not self.game_over:
                pairs = sum(1 for count in month_counts.values() if count >= 2)
                if pairs >= 4:
                    self.yaku_points[player] = 6
                    self.yaku_list[player].append("食付")
                    self.game_over = True
                    self.game_result = player
                    break
        
        return None

    def get_legal_actions_mask(self, player_id): 
        """
        生成合法动作掩码。
        """
        mask = np.zeros(38, dtype=bool)

        if self.current_player != player_id:
            return mask

        if self.turn_phase == 0:  # 0: 出牌阶段
            hand = self.player_hands[player_id]
            for play_card, hand_card in enumerate(hand):
                matching_table_cards = [card for card in self.table_cards if card.month == hand_card.month]
                if not matching_table_cards:
                    # 如果没有同月牌，只有“不配对”（选项3）是合法的
                    # 动作ID = 手牌槽位 * 4 + 配对选项
                    mask[play_card * 4 + 3] = True
                else:
                    # 如果有同月牌，可以选择配对其中任意一张
                    for match_card in range(len(matching_table_cards)):
                        mask[play_card * 4 + match_card] = True

        elif self.turn_phase == 1:  # 1: 抽牌配对阶段
            drawn_card = self.drawn_card
            matching_table_cards = [card for card in self.table_cards if card.month == drawn_card.month]
            if not matching_table_cards:
                mask[32 + 3] = True # 抽牌不配对
            else:
                for match_card in range(len(matching_table_cards)):
                    mask[32 + match_card] = True

        else:  # 2 & 3: 叫牌决策阶段
            # 动作36(不叫牌/结束) 和 37(叫牌) 都是合法的
            mask[36] = True # 结束
            mask[37] = True # 叫牌

        return mask

    def perform_action(self, action, player_id):
        """
        根据动作ID执行动作。
        """
        # 动作解析
        if self.turn_phase == 0: # 出牌阶段
            if not self.player_hands[player_id]:
                self._end_game(winner_id = -1) # 平局游戏结束
            else:
                play_card, match_choice = divmod(action, 4)
                card_to_play = self.player_hands[player_id][play_card]
                self._play_card(card_to_play, match_choice, player_id) # 出牌

        elif self.turn_phase == 1: # 抽牌配对阶段
            draw_choice = action - 32
            self._judge_draw(draw_choice, player_id) # 处理配对

        elif self.turn_phase == 2: # 出牌后叫牌阶段
            koikoi = action - 36
            self._judge_koikoi(koikoi, player_id)
            if not self.game_over:
                self._draw_card(player_id) # 进入抽牌阶段

        else: # 抽牌后叫牌阶段
            koikoi = action - 36
            self._judge_koikoi(koikoi, player_id)
            if not self.game_over:
                self._end_turn() # 进入下回合

        return None

    def _play_card(self, card_to_play, match_choice, player_id):
        self.player_hands[player_id].remove(card_to_play)
        matched_cards = [card for card in self.table_cards if card.month == card_to_play.month]

        if matched_cards and match_choice < len(matched_cards):
            collected_card = matched_cards[match_choice]
            self.table_cards.remove(collected_card)
            self.collected_cards[player_id].extend([card_to_play, collected_card])
            self._evaluate_yaku(player_id) # 检查役
            if self.turn_phase == 0:
                self._draw_card(player_id) # 抽牌
        else:
            self.table_cards.append(card_to_play)
            self.table_cards.sort(key=lambda card: card.card_id)
            self._draw_card(player_id) # 抽牌

        if not self.player_hands[player_id] and not self.player_hands[1 - player_id]:
            self._end_game(winner_id = -1) # 平局游戏结束
        
        return None

    def _draw_card(self, player_id): 
        """
        抽牌
        """
        self.drawn_card = self.draw_pile.pop()
        self.turn_phase = 1 # 进入抽牌配对阶段
        return None

    def _judge_draw(self, draw_choice, player_id):
        """
        处理抽牌逻辑：
        - 尝试与场牌配对。
        - 返回更新后的游戏状态和动作结果。
        """
        matched_cards = [card for card in self.table_cards if card.month == self.drawn_card.month]

        if matched_cards and draw_choice < len(matched_cards):
            chosen_card = matched_cards[draw_choice]
            self.table_cards.remove(chosen_card)
            self.collected_cards[player_id].extend([self.drawn_card, chosen_card])
            self._evaluate_yaku(player_id) # 检查役
            if self.turn_phase == 1:
                self._end_turn() # 进入下回合
        else:
            self.table_cards.append(self.drawn_card)
            self.table_cards.sort(key=lambda card: card.card_id)
            self._end_turn() # 进入下回合
        
        return None

    def _evaluate_yaku(self, player_id):
        """
        役判定模块：根据玩家收集的牌更新役点数和牌型。
        """
        collected_cards = self.collected_cards[player_id]
        yaku_points = 0
        self.yaku_list[player_id] = []  # 清空当前牌型

        # 统计各类牌的数量
        hikari_with_rain = [card for card in collected_cards if card.card_name == "柳间小野道风"]
        self.yaku_progress[player_id][0] = len(hikari_with_rain) / 1

        hikari_without_rain = [card for card in collected_cards if card.category == "光" and card.card_name != "柳间小野道风"]
        self.yaku_progress[player_id][1] = len(hikari_without_rain) / 4

        flower = [card for card in collected_cards if card.card_name == "樱上幕"]
        self.yaku_progress[player_id][2] = len(flower) / 1

        wine = [card for card in collected_cards if card.card_name == "菊上杯"]
        self.yaku_progress[player_id][3] = len(wine) / 1

        moon = [card for card in collected_cards if card.card_name == "芒上月"]
        self.yaku_progress[player_id][4] = len(moon) / 1

        animal = [card for card in collected_cards if card.card_name in ["萩间野猪", "枫间鹿", "牡丹上蝶"]]
        self.yaku_progress[player_id][5] = len(animal) / 3

        red_tan = [card for card in collected_cards if card.card_name in ["松上赤短", "梅上赤短", "樱上赤短"]]
        self.yaku_progress[player_id][6] = len(red_tan) / 3

        blue_tan = [card for card in collected_cards if card.card_name in ["牡丹青短", "菊上青短", "枫上青短"]]
        self.yaku_progress[player_id][7] = len(blue_tan) / 3

        tan = [card for card in collected_cards if card.category == "短册"]
        self.yaku_progress[player_id][8] = len(tan) / 5

        tane = [card for card in collected_cards if card.category == "种"]
        self.yaku_progress[player_id][9] = len(tane) / 5

        kasu = [card for card in collected_cards if card.category == "佳士"]
        self.yaku_progress[player_id][10] = len(kasu) / 10

        # 五光
        if len(hikari_with_rain) + len(hikari_without_rain) == 5:
            yaku_points += 10.
            self.yaku_list[player_id].append("五光")

        # 四光
        elif len(hikari_without_rain) == 4:
            yaku_points += 8.
            self.yaku_list[player_id].append("四光")

        # 雨四光
        elif len(hikari_without_rain) == 3 and len(hikari_with_rain) == 1:
            yaku_points += 7.
            self.yaku_list[player_id].append("雨四光")

        # 三光
        elif len(hikari_without_rain) == 3:
            yaku_points += 6.
            self.yaku_list[player_id].append("三光")

        # 花见酒（樱上幕 + 菊上杯）
        if len(flower) + len(wine) == 2:
            yaku_points += 5.
            self.yaku_list[player_id].append("花见酒")

        # 月见酒（芒上月 + 菊上杯）
        if len(moon) + len(wine) == 2:
            yaku_points += 5.
            self.yaku_list[player_id].append("月见酒")

        # 猪鹿蝶
        if len(animal) == 3:
            yaku_points += 5.
            self.yaku_list[player_id].append("猪鹿蝶")

        # 赤短
        if len(red_tan) == 3:
            yaku_points += 5.
            self.yaku_list[player_id].append("赤短")

        # 青短
        if len(blue_tan) == 3:
            yaku_points += 5.
            self.yaku_list[player_id].append("青短")

        # 短册（基础5张1分，每多1张加1分）
        if len(tan) >= 5:
            yaku_points += 1. + (len(tan) - 5)
            self.yaku_list[player_id].append(f"短册 x{1 + (len(tan) - 5)}")

        # 种（基础5张1分，每多1张加1分）
        if len(tane) >= 5:
            yaku_points += 1. + (len(tane) - 5)
            self.yaku_list[player_id].append(f"种 x{1 + (len(tane) - 5)}")

        # 佳士（基础10张1分，每多1张加1分）
        if len(kasu) >= 10:
            yaku_points += 1. + (len(kasu) - 10)
            self.yaku_list[player_id].append(f"佳士 x{1 + (len(kasu) - 10)}")

        if yaku_points > self.yaku_points[player_id]:
            self.turn_phase += 2
            self.yaku_points[player_id] = yaku_points

        return None
    
    def _judge_koikoi(self, koikoi, player_id):
        """
        判断是否叫牌
        """
        if not koikoi:
            self.koikoi_flags[player_id] = 0
            self._end_game(winner_id=player_id) # 结束游戏
        else:
            self.koikoi_flags[player_id] = 1

        return None
    
    def _end_game(self, winner_id):
        """
        判断游戏结束
        """
        self.game_over = True
        self.game_result = winner_id

        return None

    def _end_turn(self):
        """
        结束回合，切换玩家
        """
        self.current_player = 1 - self.current_player
        self.turn_phase = 0 # 进入出牌阶段
        self.drawn_card = None # 重置抽牌
        
        return None