#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
import robosuite as suite

# 保存路径
save_dir = "/root/userdata/vcsptptjdl2g/eb_data/eb_code_ws/cjx/robosuite-master/robosuite/demos/images/nut_assembly_images"
os.makedirs(save_dir, exist_ok=True)

# 创建环境 NutAssembly
env = suite.make(
    env_name="NutAssembly",
    robots="UR5e",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="frontview",  # 使用 agentview 视角
    camera_names="frontview",
    control_freq=20,
    camera_heights=1024,
    camera_widths=1024,
)

# 获取初始观测
obs = env.reset()

# 验证并保存第一帧图像
if "frontview_image" in obs:
    img_array = obs["frontview_image"]
    
    # 根据原版环境经验，尝试 180 度翻转 (如果不正可以修改这里)
    img = Image.fromarray(img_array).rotate(180, expand=True) 
    
    save_path = os.path.join(save_dir, "nut_assembly_front_snapshot.png")
    img.save(save_path)
    print(f"✅ 成功！NutAssembly 场景快照已保存至: {save_path}")
else:
    print("❌ 失败！未能获取到 agentview_image。")

env.close()
