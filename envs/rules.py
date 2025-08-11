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
        self.yaku_list = {0: [], 1: []}  # 玩家0和玩家1的役种列表
        self.current_player = 0  # 当前玩家（0或1）
        self.turn_phase = 0  # 当前阶段（0：出牌阶段，1：抽牌阶段，2：叫牌阶段）
        self.game_over = False  # 游戏是否结束

    def reset(self, np_random = None):
        """
        重置游戏状态，返回初始发牌结果。
        同时检查手牌是否符合"手四"或"食付"条件。
        """
        self.player_hands, self.table_cards, self.draw_pile = self.deck.deal(np_random)
        self.collected_cards = {0: [], 1: []}
        self.yaku_points = {0: 0, 1: 0}
        self.yaku_list = {0: [], 1: []}
        self.current_player = 0
        
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
                    break
            
            # 食付：4对2张同月份
            if not self.game_over:
                pairs = sum(1 for count in month_counts.values() if count >= 2)
                if pairs >= 4:
                    self.yaku_points[player] = 6
                    self.yaku_list[player].append("食付")
                    self.game_over = True
                    break
        
        return None

    def play_card(self, card_id, play_choice):
        """
        处理玩家出牌逻辑：
        - 玩家从手牌中选择一张牌，尝试与场牌配对。
        - 返回更新后的游戏状态和动作结果。
        """
        self.turn_phase = 0

        player = self.current_player

        # 找到玩家手牌中的目标牌
        card_to_play = None
        for card in self.player_hands[player]:
            if card.card_id == card_id:
                card_to_play = card
                break

        if card_to_play is None:
            raise ValueError("Card not found in player's hand.")

        # 从手牌中移除该牌
        self.player_hands[player].remove(card_to_play)

        # 检查场牌中是否有相同月份的牌
        matched_cards = [
            card for card in self.table_cards
            if card.month == card_to_play.month
        ]

        if matched_cards:
            if play_choice >= len(matched_cards):
                raise ValueError("Invalid play choice for matched cards.")

            # 将配对的牌加入收集区
            collected_card = matched_cards[play_choice]
            self.collected_cards[player].extend([card_to_play, collected_card])

            # 从场牌中移除配对的牌
            self.table_cards.remove(collected_card)

            # 检查是否形成役
            self.evaluate_yaku()

        else:
            if play_choice != 3:
                raise ValueError("Invalid play choice for unmatched cards.")
            
            # 没有配对，将牌加入场牌
            self.table_cards.append(card_to_play)
            # 对场牌按card_id排序
            self.table_cards = sorted(self.table_cards, key=lambda card: card.card_id)

        return None

    def draw_card(self): 
        """
        抽牌
        """
        self.turn_phase = 1
        
        self.drawn_card = self.draw_pile.pop()
        return None

    def judge_draw_card(self, draw_choice):
        """
        处理抽牌逻辑：
        - 尝试与场牌配对。
        - 返回更新后的游戏状态和动作结果。
        """
        player = self.current_player

        matched_cards = [
            card for card in self.table_cards
            if card.month == self.drawn_card.month
        ]

        if matched_cards:
            # 如果有多张匹配的牌
            if draw_choice >= len(matched_cards):
                raise ValueError("Invalid draw choice for matched cards.")
            
            chosen_card = matched_cards[draw_choice]
            self.collected_cards[player].extend([self.drawn_card, chosen_card])
            self.table_cards.remove(chosen_card)

            # 检查是否形成役
            self.evaluate_yaku()
        else:
            if draw_choice != 3:
                raise ValueError("Invalid draw choice for unmatched cards.")
            
            self.table_cards.append(self.drawn_card)
            # 对场牌按card_id排序
            self.table_cards = sorted(self.table_cards, key=lambda card: card.card_id)

        return None

    def evaluate_yaku(self):
        """
        役判定模块：根据玩家收集的牌更新役点数和牌型。
        """
        player = self.current_player

        collected_cards = self.collected_cards[player]
        yaku_points = 0
        self.yaku_list[player] = []  # 清空当前牌型

        # 统计各类牌的数量
        hikari_with_rain = [card for card in collected_cards if card.card_name == "柳间小野道风"]
        hikari_without_rain = [card for card in collected_cards if card.category == "光" and card.card_name != "柳间小野道风"]
        flower = [card for card in collected_cards if card.card_name == "樱上幕"]
        wine = [card for card in collected_cards if card.card_name == "菊上杯"]
        moon = [card for card in collected_cards if card.card_name == "芒上月"]
        animal = [card for card in collected_cards if card.card_name in ["萩间野猪", "枫间鹿", "牡丹上蝶"]]
        red_tan = [card for card in collected_cards if card.card_name in ["松上赤短", "梅上赤短", "樱上赤短"]]
        blue_tan = [card for card in collected_cards if card.card_name in ["牡丹青短", "菊上青短", "枫上青短"]]
        tan = [card for card in collected_cards if card.category == "短册"]
        tane = [card for card in collected_cards if card.category == "种"]
        kasu = [card for card in collected_cards if card.category == "佳士"]

        # 五光
        if len(hikari_with_rain) + len(hikari_without_rain) == 5:
            yaku_points += 10
            self.yaku_list[player].append("五光")

        # 四光
        elif len(hikari_without_rain) == 4:
            yaku_points += 8
            self.yaku_list[player].append("四光")

        # 雨四光
        elif len(hikari_without_rain) == 3 and len(hikari_with_rain) == 1:
            yaku_points += 7
            self.yaku_list[player].append("雨四光")

        # 三光
        elif len(hikari_without_rain) == 3:
            yaku_points += 6
            self.yaku_list[player].append("三光")

        # 花见酒（樱上幕 + 菊上杯）
        if len(flower) + len(wine) == 2:
            yaku_points += 5
            self.yaku_list[player].append("花见酒")

        # 月见酒（芒上月 + 菊上杯）
        if len(moon) + len(wine) == 2:
            yaku_points += 5
            self.yaku_list[player].append("月见酒")

        # 猪鹿蝶
        if len(animal) == 3:
            yaku_points += 5
            self.yaku_list[player].append("猪鹿蝶")

        # 赤短
        if len(red_tan) == 3:
            yaku_points += 5
            self.yaku_list[player].append("赤短")

        # 青短
        if len(blue_tan) == 3:
            yaku_points += 5
            self.yaku_list[player].append("青短")

        # 短册（基础5张1分，每多1张加1分）
        if len(tan) >= 5:
            yaku_points += 1 + (len(tan) - 5)
            self.yaku_list[player].append(f"短册 x{1 + (len(tan) - 5)}")

        # 种（基础5张1分，每多1张加1分）
        if len(tane) >= 5:
            yaku_points += 1 + (len(tane) - 5)
            self.yaku_list[player].append(f"种 x{1 + (len(tane) - 5)}")

        # 佳士（基础10张1分，每多1张加1分）
        if len(kasu) >= 10:
            yaku_points += 1 + (len(kasu) - 10)
            self.yaku_list[player].append(f"佳士 x{1 + (len(kasu) - 10)}")

        if yaku_points > self.yaku_points[player]:
            self.turn_phase = 2
            self.yaku_points[player] = yaku_points

        return None
    
    def judge_koikoi(self, koikoi):
        """
        判断是否叫牌
        """
        if not koikoi:
            self.game_over = True

        return None
    
    def switch_player(self):
        """
        切换玩家
        """
        self.current_player = 1 - self.current_player
        
        return None