"""
对 HanafudaEnv 的完整测试套件。

此测试套件验证以下内容：
1.  环境是否遵循 Gymnasium API (使用 stable-baselines3 的 check_env)。
2.  reset 和 step 函数是否返回正确的数据结构和类型。
3.  环境是否能正确处理非法动作。
4.  在所有游戏阶段（出牌、抽牌、叫牌），动作掩码（action_mask）是否生成正确。
5.  在大量的随机游戏中，环境是否能保持稳定而不崩溃。
"""

import pytest
import numpy as np

# 假设你的项目结构是 hanafuda_rl/，并且你的环境代码在 hanafuda_rl/envs/ 中
# 如果不是，请相应地调整导入路径
from hanafuda_rl.envs.hanafuda_env import HanafudaEnv
from hanafuda_rl.envs.rules import Deck

# 使用 stable-baselines3 的环境检查器，这是黄金标准
from stable_baselines3.common.env_checker import check_env


# Pytest Fixture: 为每个测试函数提供一个干净、初始化的环境实例
@pytest.fixture
def env():
    """提供一个 HanafudaEnv 的实例。"""
    return HanafudaEnv()

# Pytest Fixture: 提供一个固定的、可复现的牌组实例，用于手动设置场景
@pytest.fixture
def deck():
    """提供一个标准牌组实例，用于获取特定卡牌。"""
    return Deck()


# --- 测试 1: API 合规性 ---
def test_env_api_compliance(env):
    """
    黄金标准测试：使用 SB3 的 check_env 验证环境是否符合标准 API。
    如果此测试通过，你的环境与 SB3/SB3-Contrib 库的兼容性就有保障了。
    """
    # warn=True 会打印警告而不是直接抛出异常，对于调试更友好
    # 对于最终测试，可以移除 warn=True
    check_env(env, warn=True)


# --- 测试 2: 核心功能 (Reset & Step) ---
def test_reset_functionality(env):
    """验证 reset() 方法的返回值是否正确。"""
    # 使用固定的种子以确保测试的可复现性
    obs, info = env.reset(seed=42)

    # 1. 验证 Observation
    assert isinstance(obs, dict), "Observation 应该是一个字典"
    assert set(obs.keys()) == set(env.observation_space.spaces.keys()), "Observation 的键与 observation_space 不匹配"
    assert obs['hand'].shape == (48,), "手牌观测形状错误"
    assert obs['hand'].dtype == np.int8, "手牌观测数据类型错误"
    
    # 2. 验证 Info 字典
    assert isinstance(info, dict), "Info 应该是一个字典"
    assert 'action_mask' in info, "Info 字典中必须包含 'action_mask'"
    
    # 3. 验证 Action Mask
    mask = info['action_mask']
    assert isinstance(mask, np.ndarray), "Action mask 应该是 numpy 数组"
    assert mask.shape == (38,), "Action mask 的形状应该是 (38,)"
    assert mask.dtype == bool, "Action mask 的数据类型应该是布尔型"


def test_step_functionality_and_illegal_action(env):
    """验证 step() 方法的功能，并测试非法动作是否会按预期引发异常。"""
    obs, info = env.reset(seed=42)
    mask = info['action_mask']

    # 1. 测试合法动作
    legal_actions = np.where(mask)[0]
    assert len(legal_actions) > 0, "在游戏开始时不应没有任何合法动作"
    action = legal_actions[0]

    # 执行一个合法动作
    new_obs, reward, terminated, truncated, new_info = env.step(action)

    # 验证返回值类型
    assert isinstance(new_obs, dict)
    assert isinstance(reward, float)  # Gym API 推荐奖励为 float 类型
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(new_info, dict)
    assert 'action_mask' in new_info, "每次 step 后，info 中都必须有 action_mask"


# --- 测试 3: 动作掩码的正确性 (关键测试) ---

def test_action_mask_play_phase(env, deck):
    """精确验证在'出牌阶段'的动作掩码逻辑。"""
    env.reset(seed=1) # 初始化内部状态
    env._turn_phase = 0 # 强制设置为出牌阶段

    # 牌的定义 (ID, 月份)
    matsu_hikari = deck.cards[0]  # 松(1), 光
    matsu_tan = deck.cards[1]     # 松(1), 短册
    ume_kasu = deck.cards[6]      # 梅(2), 佳士
    
    # 场景A: 手牌有一张牌，场上没有匹配项
    env.rules.player_hands[0] = [matsu_hikari]
    env.rules.table_cards = [ume_kasu]
    mask_a = env.get_action_mask()
    expected_a = np.zeros(38, dtype=bool)
    expected_a[0 * 4 + 3] = True # 第0张手牌，选择“不配对”(选项3)
    np.testing.assert_array_equal(mask_a, expected_a, "场景A：无匹配时的掩码错误")

    # 场景B: 手牌有一张牌，场上有一张匹配项
    env.rules.player_hands[0] = [matsu_hikari]
    env.rules.table_cards = [matsu_tan]
    mask_b = env.get_action_mask()
    expected_b = np.zeros(38, dtype=bool)
    expected_b[0 * 4 + 0] = True # 第0张手牌，选择配对第0张场牌
    np.testing.assert_array_equal(mask_b, expected_b, "场景B：有唯一匹配时的掩码错误")


def test_action_mask_draw_phase(env, deck):
    """精确验证在'抽牌配对阶段'的动作掩码逻辑。"""
    env.reset(seed=1)
    env._turn_phase = 1 # 强制设置为抽牌配对阶段

    # 牌的定义
    matsu_hikari = deck.cards[0]
    matsu_tan = deck.cards[1]
    ume_kasu = deck.cards[6]
    
    # 场景A: 抽中的牌在场上没有匹配项
    env.rules.drawn_card = matsu_hikari # 手动设置抽中的牌
    env.rules.table_cards = [ume_kasu]
    mask_a = env.get_action_mask()
    expected_a = np.zeros(38, dtype=bool)
    expected_a[32 + 3] = True # 抽牌动作(32-35)，选择“不配对”(选项3)
    np.testing.assert_array_equal(mask_a, expected_a, "抽牌场景A：无匹配时的掩码错误")

    # 场景B: 抽中的牌在场上有一张匹配项
    env.rules.drawn_card = matsu_hikari
    env.rules.table_cards = [matsu_tan]
    mask_b = env.get_action_mask()
    expected_b = np.zeros(38, dtype=bool)
    expected_b[32 + 0] = True # 抽牌动作，选择配对第0张场牌
    np.testing.assert_array_equal(mask_b, expected_b, "抽牌场景B：有唯一匹配时的掩码错误")


# 使用 parametrize 可以用一个函数测试多种情况
@pytest.mark.parametrize("phase", [2, 3])
def test_action_mask_koikoi_phase(env, phase):
    """精确验证在'叫牌决策阶段'的动作掩码逻辑。"""
    env.reset(seed=1)
    env._turn_phase = phase # 设置为出牌后(2)或抽牌后(3)的叫牌阶段

    mask = env.get_action_mask()
    expected = np.zeros(38, dtype=bool)
    expected[36] = True # 不叫牌(Stop)
    expected[37] = True # 叫牌(Koi-Koi)
    
    np.testing.assert_array_equal(mask, expected, f"叫牌阶段 {phase} 的掩码错误")


# --- 测试 4: 随机运行稳定性 ---
@pytest.mark.long # 标记为长时间测试，可以通过 pytest -m "not long" 跳过
def test_random_rollout_stability(env):
    """
    通过运行大量随机对局来对环境进行压力测试。
    目的是捕捉那些在特定代码路径下才会出现的边缘案例、崩溃或死锁。
    """
    num_episodes = 100000  # 可以增加到1000进行更彻底的本地测试

    for i in range(num_episodes):
        obs, info = env.reset(seed=i) # 使用不同的种子进行多样化测试
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            action_mask = info['action_mask']
            
            # 这是最重要的断言：在游戏结束前，必须总是有至少一个合法动作。
            # 如果这里失败，说明你的状态机或掩码生成在某个边缘情况下出错了。
            assert np.any(action_mask), f"在第 {i+1} 局, 第 {step_count} 步出现死锁! 没有任何合法动作. 阶段: {obs['turn_phase']}"
            
            # 从合法动作中随机选择一个
            # 使用 env.np_random 以尊重环境的随机种子
            legal_actions = np.where(action_mask)[0]
            action = env.np_random.choice(legal_actions)
            
            # 执行动作，我们只关心它是否会崩溃
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            # 防止无限循环的保护
            assert step_count < 200, f"第 {i+1} 局游戏超过200步，可能存在无限循环"