"""
Driver class for VR controller using OSC (Operational Space Control).
"""

import numpy as np
from threading import Thread

from robosuite.devices.device import Device

# Try to import ROS2 and XR messages
try:
    import rclpy
    from rclpy.node import Node
    from xr_msgs.msg import Custom
    ROS_AVAILABLE = True
except ImportError as e:
    print(f"ROS2 or XR messages import error: {e}")
    ROS_AVAILABLE = False

# Global variables to store VR controller deltas and button states
goal = np.array([0.0, 0.0, 0.0])
goal_2 = np.array([0.0, 0.0, 0.0])
left_trigger_pressed = False
right_trigger_pressed = False


def second_print(msg, interval=1.0):
    """Simple print function"""
    print(msg)


if ROS_AVAILABLE:
    class XrSubscriber(Node):
        def __init__(self):
            super().__init__('xr_subscriber')
            print("=== Creating XR subscriber ===")
            self.subscription = self.create_subscription(
                Custom,
                'xr_pose',
                self.xr_callback,
                10)
            self._cnt = 0
            self.origin = None
            self.origin_2 = None
            print("=== XR subscriber created successfully ===")

        def xr_callback(self, msg):
            global goal, goal_2, left_trigger_pressed, right_trigger_pressed
            
            # Left controller
            if msg.left_controller.status == 3 and msg.left_controller.pose is not None:
                pose = msg.left_controller.pose
                
                # 获取trigger值
                left_trigger_pressed = msg.left_controller.trigger > 0.5
                
                if self._cnt == 0:
                    self.origin = pose
                    print(f"Set left origin: {self.origin}")

                delta_xr = pose[:3] - self.origin[:3]
                x, y, z = delta_xr
                
                # 坐标转换 - 修复前后和左右方向
                # VR: x(右), y(上), z(后)
                # 修复:
                # VR右(x) -> MJ左(y)  (反转左右)
                # VR上(y) -> MJ上(z)  (上下正确)
                # VR前(z) -> MJ后(-x) (反转前后)
                delta_mj = np.array([-z, x, y])
                
                goal = delta_mj
                
                if self._cnt % 30 == 0:
                    print(f"VR delta: [{x:.3f}, {y:.3f}, {z:.3f}] -> MJ: {delta_mj}, trigger: {left_trigger_pressed}")

            # Right controller
            if msg.right_controller.status == 3 and msg.right_controller.pose is not None:
                pose = msg.right_controller.pose
                right_trigger_pressed = msg.right_controller.trigger > 0.5
                
                if self._cnt == 0:
                    self.origin_2 = pose
                    print(f"Set right origin: {self.origin_2}")

                delta_xr = pose[:3] - self.origin_2[:3]
                x, y, z = delta_xr
                delta_mj = np.array([-z, x, y])
                goal_2 = delta_mj

            self._cnt += 1


def run_ros_node():
    """Run ROS2 node in separate thread"""
    print("=== Starting ROS2 node ===")
    try:
        rclpy.init()
        subscriber = XrSubscriber()
        rclpy.spin(subscriber)
    except Exception as e:
        print(f"ROS2 node error: {e}")
    finally:
        if 'subscriber' in locals():
            subscriber.destroy_node()
        rclpy.shutdown()


class VR_OSC(Device):
    """VR controller device using Operational Space Control."""

    def __init__(self, env, pos_sensitivity=8.0, rot_sensitivity=1.0):
        super().__init__(env)
        
        self._display_controls()
        
        self._reset_state = 0
        self._enabled = False
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        
        self.vr_delta = np.zeros(3)
        self.last_vr_delta = np.zeros(3)
        
        self.prev_grasp = False
        self.grasp_toggle_cooldown = 0
        
        self.ros_thread = None

    @staticmethod
    def _display_controls():
        """Display control instructions"""
        print("\nVR Controls (OSC Mode)")
        print("-" * 40)
        print("Left Controller    Move end effector")
        print("Left Trigger       Toggle gripper")
        print("B Button           Reset simulation")
        print("-" * 40 + "\n")

    def _reset_internal_state(self):
        """Reset internal state"""
        super()._reset_internal_state()
        
        self.rotation = np.array([[-1.0, 0.0, 0.0], 
                                  [0.0, 1.0, 0.0], 
                                  [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)
        self.last_pos = np.zeros(3)
        
        self.vr_delta = np.zeros(3)
        self.last_vr_delta = np.zeros(3)
        
        self.prev_grasp = False
        self.grasp_toggle_cooldown = 0

    def start_control(self):
        """Start VR control"""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        
        print("VR_OSC control started")
        print(f"Position sensitivity: {self.pos_sensitivity}")
        
        if ROS_AVAILABLE:
            print("Starting ROS2 thread...")
            self.ros_thread = Thread(target=run_ros_node)
            self.ros_thread.daemon = True
            self.ros_thread.start()
        else:
            print("ROS2 not available!")

    def get_controller_state(self):
        """Get current VR controller state"""
        global goal, left_trigger_pressed
        self.vr_delta = goal.copy()

        dpos = self.vr_delta - self.last_vr_delta
        self.last_vr_delta = self.vr_delta.copy()

        current_trigger = bool(left_trigger_pressed)
        if current_trigger and (not self.prev_grasp) and self.grasp_toggle_cooldown == 0:
            self.toggle_grasp()
            self.grasp_toggle_cooldown = 10
        self.prev_grasp = current_trigger
        if self.grasp_toggle_cooldown > 0:
            self.grasp_toggle_cooldown -= 1

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.zeros(3),
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def toggle_grasp(self):
        """Toggle gripper state"""
        current_grasp = self.grasp_states[self.active_robot][self.active_arm_index]
        self.grasp_states[self.active_robot][self.active_arm_index] = not current_grasp
        new_grasp = self.grasp_states[self.active_robot][self.active_arm_index]
        print(f"Grasp toggled: {current_grasp} -> {new_grasp}")

    def reset(self):
        """Reset controller"""
        self._reset_state = 1
        self._enabled = False
        self._reset_internal_state()
        print("VR_OSC reset")

    def _postprocess_device_outputs(self, dpos, drotation):
        """Scale and clip outputs"""
        dpos = dpos * self.pos_sensitivity
        drotation = drotation * self.rot_sensitivity
        
        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
