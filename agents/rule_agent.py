class RuleAgent:
    def __init__(self):
        pass

    def select_action(self, observation, action_mask):
        if any(action_mask[:32]):
            for idx in [0, 4, 8, 12, 16, 20, 24, 28]:
                if action_mask[idx]:
                    return idx
            return 3
        elif any(action_mask[32:36]):
            for idx in range(32,36):
                if action_mask[idx]:
                    return idx
        else:
            return 36