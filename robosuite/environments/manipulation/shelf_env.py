import numpy as np
from collections import OrderedDict

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# 导入我们自定义的 ShelfArena
try:
    from robosuite.models.arenas.shelf_arena import ShelfArena
except ImportError:
    from robosuite.models.arenas.shelf_arena import ShelfArena

# 导入 BoxObject (用于创建单个盒子实例)
from robosuite.models.objects import BoxObject

class ShelfEnv(ManipulationEnv):
    """
    自定义货架环境：机器人在货架前整理/抓取盒子。
    仿照 robosuite.environments.manipulation.lift.Lift 编写。
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="sideview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="sideview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        # --- 自定义参数 ---
        shelf_pos=(0, 0.4, 0),
        num_boxes=3,
        # 盒子使用 half-size: x / y 控制长宽，z 单独控制高度
        # 默认做成更窄、更高的长方体，侧面不接近正方形，便于二指夹爪抓取
        box_size_range=([0.02, 0.02, 0.05], [0.035, 0.035, 0.05]),
        # 盒子默认放在机械臂前方的地面区域，范围稍大以便拉开彼此距离
        box_placement_range=[-0.24, 0.24, -0.06, 0.24], # x_min, x_max, y_min, y_max
        z_offset=0.1,
    ):
        # 保存自定义参数
        self.shelf_pos = shelf_pos
        self.num_boxes = num_boxes
        self.box_size_range = box_size_range
        self.box_placement_range = box_placement_range
        self.z_offset = z_offset
        
        # 奖励配置
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        
        # 初始化盒子列表 (将在 _load_model 中具体实例化)
        self.boxes = []
        self.placement_initializer = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        简单的奖励函数示例：如果任何盒子被抓起并抬高，给予奖励。
        这里仅作演示，可根据任务需求修改。
        """
        reward = 0.0
        # 简单示例：检查第一个盒子是否被抬起
        if len(self.boxes) > 0:
            box_id = self.sim.model.body_name2id(self.boxes[0].root_body)
            box_z = self.sim.data.body_xpos[box_id][2]
            # 假设货架高度约为 0.8，抬起阈值为 1.0
            if box_z > 1.0:
                reward = 1.0
        
        if self.reward_scale is not None:
            reward *= self.reward_scale
            
        return reward

    def _load_model(self):
        """
        加载模型：ShelfArena + Robots + Boxes
        仿照 Lift._load_model
        """
        super()._load_model()

        # 1. 调整机器人基座位置 (保持不变)
        for robot in self.robots:
            robot.robot_model.set_base_xpos([0, -0.5, 0]) 

        # 2. 初始化 Arena (保持不变)
        mujoco_arena = ShelfArena(
            shelf_pos=self.shelf_pos,
            xml="arenas/shelf_arena.xml"
        )
        mujoco_arena.set_origin([0, 0, 0])

        # 3. 初始化对象 (Boxes) (保持不变)
        self.boxes = []
        min_size, max_size = self.box_size_range
        
        tex_attrib = {"type": "cube"}
        mat_attrib = {"specular": "0.4", "shininess": "0.1"}
        blue_mat = CustomMaterial(
            texture="WoodBlue", 
            tex_name="bluewood", 
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib, 
            mat_attrib=mat_attrib
        )

        for i in range(self.num_boxes):
            size = np.random.uniform(min_size, max_size)
            box = BoxObject(
                name=f"box_{i}",
                size_min=size, 
                size_max=size, 
                rgba=[np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0), 0.2, 1.0],
                material=blue_mat,
                rng=self.rng
            )
            self.boxes.append(box)

        # 4. 设置放置初始化器 (保持不变，虽然_reset_internal会覆盖它，但保留以防万一)
        x_min, x_max, y_min, y_max = self.box_placement_range
        
        self.placement_initializer = UniformRandomSampler(
            name="BoxSampler",
            mujoco_objects=self.boxes,
            x_range=[x_min, x_max],
            y_range=[y_min, y_max],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=np.array([0, 0, 0.0]),
            z_offset=self.z_offset,
            rng=self.rng,
        )

        # 5. 构建 Task (保持不变)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.boxes,
        )
        
        # 【仅修改打印信息】提示新的初始摆放策略
        print(f"✅ ShelfEnv 模型加载完成: {len(self.boxes)} 个盒子")
        print(f"   -> 盒子将随机放置在机械臂前方地面区域 {self.box_placement_range}")

    def _setup_references(self):
        """
        设置引用 ID (保持不变)
        """
        super()._setup_references()
        
        self.box_body_ids = []
        for box in self.boxes:
            try:
                bid = self.sim.model.body_name2id(box.root_body)
                self.box_body_ids.append(bid)
            except Exception as e:
                print(f"警告：无法找到盒子 {box.name} 的 ID: {e}")
                self.box_body_ids.append(None)

    def _setup_observables(self):
        """
        设置观测值 (保持不变)
        """
        observables = super()._setup_observables()
        
        if self.use_object_obs and len(self.boxes) > 0:
            modality = "object"
            box_idx = 0
            box_name = self.boxes[box_idx].name
            
            @sensor(modality=modality)
            def box_pos(obs_cache):
                if self.box_body_ids[box_idx] is not None:
                    return np.array(self.sim.data.body_xpos[self.box_body_ids[box_idx]])
                return np.zeros(3)

            sensors = [box_pos]
            names = [f"{box_name}_pos"]
            
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
                
        return observables

    def _reset_internal(self):
        """
        重置内部状态：将盒子放在机械臂前方地面区域，方便抓取后再放上货架
        """
        super()._reset_internal()

        if not hasattr(self, 'boxes') or len(self.boxes) == 0:
            print("❌ 错误：self.boxes 为空或未定义。")
            return
        
        target_objects = self.boxes

        x_min, x_max, y_min, y_max = self.box_placement_range
        ground_z = 0.0
        safety_margin = 0.01
        inter_box_margin = 0.08
        placed_boxes = []

        for i, box in enumerate(target_objects):
            if hasattr(box, 'size'):
                if isinstance(box.size, (list, np.ndarray)):
                    current_half_x = float(box.size[0])
                    current_half_y = float(box.size[1])
                    current_half_height = float(box.size[2])
                else:
                    current_half_x = float(box.size)
                    current_half_height = float(box.size)
                    current_half_y = float(box.size)
            else:
                current_half_x = 0.02
                current_half_height = 0.02
                current_half_y = 0.02

            safe_x_min = x_min + current_half_x + safety_margin
            safe_x_max = x_max - current_half_x - safety_margin
            safe_y_min = y_min + current_half_y + safety_margin
            safe_y_max = y_max - current_half_y - safety_margin

            placed = False
            x_pos, y_pos = 0.0, 0.0
            for _ in range(300):
                if safe_x_min >= safe_x_max or safe_y_min >= safe_y_max:
                    cand_x, cand_y = 0.0, 0.0
                else:
                    cand_x = float(self.rng.uniform(safe_x_min, safe_x_max))
                    cand_y = float(self.rng.uniform(safe_y_min, safe_y_max))

                valid = True
                for prev_x, prev_y, prev_half_x, prev_half_y in placed_boxes:
                    overlap_x = abs(cand_x - prev_x) < (current_half_x + prev_half_x + inter_box_margin)
                    overlap_y = abs(cand_y - prev_y) < (current_half_y + prev_half_y + inter_box_margin)
                    if overlap_x and overlap_y:
                        valid = False
                        break

                if valid:
                    x_pos, y_pos = cand_x, cand_y
                    placed_boxes.append((x_pos, y_pos, current_half_x, current_half_y))
                    placed = True
                    break

            if not placed:
                x_pos, y_pos = 0.0, 0.0
                placed_boxes.append((x_pos, y_pos, current_half_x, current_half_y))

            z_pos = 0.5
            target_pos = np.array([x_pos, y_pos, z_pos])
            target_quat = np.array([1, 0, 0, 0])

            if len(box.joints) > 0:
                try:
                    addr = self.sim.model.get_joint_qpos_addr(box.joints[0])
                    
                    if isinstance(addr, tuple):
                        start_idx, end_idx = addr
                        if end_idx - start_idx >= 7:
                            self.sim.data.qpos[start_idx : start_idx+3] = target_pos
                            self.sim.data.qpos[start_idx+3 : start_idx+7] = target_quat
                    else:
                        jid = int(addr)
                        self.sim.data.qpos[jid : jid+3] = target_pos
                        self.sim.data.qpos[jid+3 : jid+7] = target_quat
                    
                    # 调试打印 (打印所有)
                    print(f"  📦 盒子 {i}: 尺寸(W={current_half_x:.3f}, D={current_half_y:.3f}, H={current_half_height:.3f}) | "
                          f"目标 Pos=[{x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}] | "
                          f"地面摆放")

                except Exception as e:
                    print(f"⚠️ 设置盒子 {i} 位置失败: {e}")
            else:
                print(f"⚠️ 盒子 {i} 没有关节，无法设置位置。")
    def _check_success(self):
        """
        检查任务是否成功 (示例：第一个盒子是否高于某个高度)
        """
        if len(self.box_body_ids) == 0 or self.box_body_ids[0] is None:
            return False
            
        box_z = self.sim.data.body_xpos[self.box_body_ids[0]][2]
        # 阈值可根据实际情况调整
        return box_z > 1.2

    def visualize(self, vis_settings):
        """
        可视化辅助 (可选)
        """
        super().visualize(vis_settings)
