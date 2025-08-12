from sb3_contrib import MaskablePPO

class PPOAgent:
    """一个包装了已训练的 PPO 模型的智能体。"""
    def __init__(self, model_path):
        try:
            self.model = MaskablePPO.load(model_path, device="cpu")
            print(f"PPO model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model from '{model_path}': {e}")
            raise

    def select_action(self, observation, action_mask):
        action, _ = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=True
        )
        return int(action)