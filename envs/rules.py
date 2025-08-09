from typing import List, Dict, Tuple, Optional
import random


class Card:
    """
    花札牌类，表示一张花札牌。
    """

    def __init__(self, card_id: int, month: int, category: str, points: int, card_name: str):
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

    def _initialize_deck(self) -> List[Card]:
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

    def deal(self, seed: Optional[int] = None) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        发牌逻辑：
        - 返回玩家手牌（2×8张）、场牌（8张）、剩余牌堆。
        - 支持随机或确定性发牌（通过seed参数）。
        """
        if seed is not None:
            random.seed(seed)

        shuffled_cards = random.sample(self.cards, len(self.cards))

        # 玩家手牌（每人8张）
        player1_hand = shuffled_cards[:8]
        player2_hand = shuffled_cards[8:16]

        # 场牌（8张）
        table_cards = shuffled_cards[16:24]

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
        self.collected_cards = {0: [], 1: []}  # 玩家0和玩家1的收集牌
        self.yaku_points = {0: 0, 1: 0}  # 玩家0和玩家1的役种分数
        self.yaku_list = {0: [], 1: []}  # 玩家0和玩家1的役种列表
        self.current_player = 0  # 当前玩家（0或1）

    def reset(self, seed: Optional[int] = None):
        """
        重置游戏状态，返回初始发牌结果。
        同时检查手牌是否符合"手四"或"食付"条件。
        """
        self.player_hands, self.table_cards, self.draw_pile = self.deck.deal(seed)
        self.collected_cards = {0: [], 1: []}
        self.yaku_points = {0: 0, 1: 0}
        self.yaku_list = {0: [], 1: []}
        self.current_player = 0
        
        force_end = False
        
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
                    force_end = True
                    break
            
            # 食付：4对2张同月份
            if not force_end:
                pairs = sum(1 for count in month_counts.values() if count >= 2)
                if pairs >= 4:
                    self.yaku_points[player] = 6
                    self.yaku_list[player].append("食付")
                    force_end = True
                    break
        
        return None

    def play_card(self, player: int, card_id: int, table_choice: int, drawn_choice: int):
        """
        处理玩家出牌逻辑：
        - 玩家从手牌中选择一张牌，尝试与场牌配对。
        - 无论配对成功与否，都会从山牌中抽一张牌进行配对。
        - 如果抽出的牌与场上的多张牌匹配，使用drawn_choice参数选择配对的牌。
        - 返回更新后的游戏状态和动作结果。
        """
        if player != self.current_player:
            raise ValueError("Not the current player's turn.")

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
            if table_choice >= len(matched_cards):
                raise ValueError("Invalid table choice for matched cards.")

            # 将配对的牌加入收集区
            collected_card = matched_cards[table_choice]
            self.collected_cards[player].extend([card_to_play, collected_card])

            # 从场牌中移除配对的牌
            self.table_cards.remove(collected_card)

            # 检查是否形成役
            self.evaluate_yaku(player)

        else:
            # 没有配对，将牌加入场牌
            self.table_cards.append(card_to_play)

        # 抽牌配对
        if self.draw_pile:
            drawn_card = self.draw_pile.pop(0)
            drawn_matched_cards = [
                card for card in self.table_cards
                if card.month == drawn_card.month
            ]

            if drawn_matched_cards:
                # 如果有多张匹配的牌，使用drawn_choice选择配对的牌
                if drawn_choice >= len(drawn_matched_cards):
                    raise ValueError("Must provide drawn_choice when multiple cards match.")
                
                # 选择配对的牌
                chosen_card = drawn_matched_cards[drawn_choice]
                self.collected_cards[player].extend([drawn_card, chosen_card])
                self.table_cards.remove(chosen_card)
                # 检查是否形成役
                self.evaluate_yaku(player)
            else:
                self.table_cards.append(drawn_card)

        self.current_player = 1 - self.current_player

        return None

    def evaluate_yaku(self, player: int):
        """
        役判定模块：根据玩家收集的牌更新役点数和牌型。
        """
        collected_cards = self.collected_cards[player]
        self.yaku_points[player] = 0
        self.yaku_list[player] = []

        # 统计各类牌的数量
        hikari = [card for card in collected_cards if card.category == "光"]
        tane = [card for card in collected_cards if card.category == "种"]
        tan = [card for card in collected_cards if card.category == "短册"]
        kasu = [card for card in collected_cards if card.category == "佳士"]

        # 五光
        if len(hikari) == 5:
            self.yaku_points[player] += 10
            self.yaku_list[player].append("五光")

        # 四光
        elif len(hikari) == 4 and not any(card.card_name == "柳间小野道风" for card in hikari):
            self.yaku_points[player] += 8
            self.yaku_list[player].append("四光")

        # 雨四光
        elif len(hikari) == 4 and any(card.card_name == "柳间小野道风" for card in hikari):
            self.yaku_points[player] += 7
            self.yaku_list[player].append("雨四光")

        # 三光
        elif len(hikari) == 3 and not any(card.card_name == "柳间小野道风" for card in hikari):
            self.yaku_points[player] += 6
            self.yaku_list[player].append("三光")

        # 猪鹿蝶
        if any(card.card_name == "牡丹上蝶" for card in tane) and any(card.card_name == "萩间野猪" for card in tane) and any(card.card_name == "枫间鹿" for card in tane):
            self.yaku_points[player] += 5
            self.yaku_list[player].append("猪鹿蝶")

        # 赤短
        red_tan = [card for card in tan if card.card_name in ["松上赤短", "梅上赤短", "樱上赤短"]]
        if len(red_tan) == 3:
            self.yaku_points[player] += 5
            self.yaku_list[player].append("赤短")

        # 青短
        blue_tan = [card for card in tan if card.card_name in ["牡丹青短", "菊上青短", "枫上青短"]]
        if len(blue_tan) == 3:
            self.yaku_points[player] += 5
            self.yaku_list[player].append("青短")

        # 花见酒（樱上幕 + 菊上杯）
        if any(card.card_name == "樱上幕" for card in collected_cards) and any(card.card_name == "菊上杯" for card in collected_cards):
            self.yaku_points[player] += 5
            self.yaku_list[player].append("花见酒")

        # 月见酒（芒上月 + 菊上杯）
        if any(card.card_name == "芒上月" for card in collected_cards) and any(card.card_name == "菊上杯" for card in collected_cards):
            self.yaku_points[player] += 5
            self.yaku_list[player].append("月见酒")

        # 短册（基础5张1分，每多1张加1分）
        if len(tan) >= 5:
            self.yaku_points[player] += 1 + (len(tan) - 5)
            self.yaku_list[player].append(f"短册 x{1 + (len(tan) - 5)}")

        # 种（基础5张1分，每多1张加1分）
        if len(tane) >= 5:
            self.yaku_points[player] += 1 + (len(tane) - 5)
            self.yaku_list[player].append(f"种 x{1 + (len(tane) - 5)}")

        # 佳士（基础10张1分，每多1张加1分）
        if len(kasu) >= 10:
            self.yaku_points[player] += 1 + (len(kasu) - 10)
            self.yaku_list[player].append(f"佳士 x{1 + (len(kasu) - 10)}")

        return None