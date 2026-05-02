import numpy as np
# 1. 直接导入你自定义的环境类
from robosuite.environments.manipulation.shelf_env import ShelfEnv
from robosuite.models.robots.manipulators.ur5e_robot import UR5e

print("✅ 自定义 ShelfEnv 类导入成功！")

# 2. 直接实例化类，而不是用 suite.make
print("\n正在创建环境...")
env = ShelfEnv(
    robots="UR5e",        # 或者传入机器人对象 robots=[UR5e()]
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera="frontview",
    control_freq=20,
    horizon=1000,
    shelf_pos=(0, 0, 0),
    num_boxes=10
)

print("✅ 环境创建成功！")

# 3. 重置并测试
obs = env.reset()
print(f"观测值 Keys: {list(obs.keys())}")
print(f"Worldbody Keys: {list(env.model.worldbody.keys())}")

if "shelf" in env.model.worldbody:
    print("🎉 成功！货架已加载到场景中。")
else:
    print("❌ 失败！未找到货架。")

# 简单运行几步
for i in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
print("演示结束。")