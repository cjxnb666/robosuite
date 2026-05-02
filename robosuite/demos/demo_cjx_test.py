#!/usr/bin/env python3
#frontview
import os
import numpy as np
from PIL import Image
import robosuite as suite

# 保存路径
save_dir = "/root/userdata/vcsptptjdl2g/eb_data/eb_code_ws/cjx/robosuite-master/robosuite/demos/images/table_images"
os.makedirs(save_dir, exist_ok=True)

# 创建环境
env = suite.make(
    env_name="Lift",
    robots="UR5e",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="sideview",
    camera_names="sideview", # 强制返回 frontview 图像
    control_freq=20,
    camera_heights=1024,         # 输出图像高度
    camera_widths=1024,   
)

# ✅ 正确获取 action_spec
low, high = env.action_spec
action_dim = low.shape[0]

obs = env.reset()

# 验证并保存第一帧图像
if "sideview_image" in obs:
    img_array = obs["sideview_image"]
    # 顺时针旋转 90°
    img = Image.fromarray(img_array).rotate(-180, expand=True)
    save_path = os.path.join(save_dir, "table_side_snapshot.png")
    img.save(save_path)
    print(f"✅ 成功！Table 场景快照已保存至: {save_path}")
else:
    print("❌ 失败！未能获取到 sideview_image。")

env.close()
print("Done!")
print(obs.keys())