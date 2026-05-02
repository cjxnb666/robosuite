"""
A script to collect a batch of human demonstrations with explicit hotkey-based
recording control.

Compared to collect_human_demonstrations.py, this version supports:
- start recording by hotkey (default: F9)
- stop current recording by hotkey (default: F10)
- exit script by hotkey (default: F12)

The demonstrations can be played back using the
`playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np
from pynput.keyboard import Key, Listener

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


class RecordingHotkeys:
    """Simple state holder for start / stop / exit recording hotkeys."""

    def __init__(self, start_key="f9", stop_key="f10", exit_key="f12"):
        self.start_key_spec = self._parse_key_spec(start_key)
        self.stop_key_spec = self._parse_key_spec(stop_key)
        self.exit_key_spec = self._parse_key_spec(exit_key)

        self.start_requested = False
        self.stop_requested = False
        self.exit_requested = False

        self.listener = Listener(on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    @staticmethod
    def _parse_key_spec(spec):
        spec = spec.strip().lower()
        if hasattr(Key, spec):
            return ("special", getattr(Key, spec))
        if len(spec) == 1:
            return ("char", spec)
        raise ValueError(
            f"Unsupported key spec '{spec}'. Use single char keys (e.g. 'r') "
            "or pynput Key names (e.g. 'f9', 'f10', 'esc')."
        )

    @staticmethod
    def _matches(key, parsed_spec):
        kind, value = parsed_spec
        if kind == "special":
            return key == value

        key_char = getattr(key, "char", None)
        if key_char is None:
            return False
        return key_char.lower() == value

    def _on_release(self, key):
        if self._matches(key, self.exit_key_spec):
            self.exit_requested = True
            return
        if self._matches(key, self.start_key_spec):
            self.start_requested = True
            return
        if self._matches(key, self.stop_key_spec):
            self.stop_requested = True

    def consume_start(self):
        if self.start_requested:
            self.start_requested = False
            return True
        return False

    def consume_stop(self):
        if self.stop_requested:
            self.stop_requested = False
            return True
        return False

    def close(self):
        self.listener.stop()



def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode, hotkeys):
    """
    Use the selected device to collect one demonstration trajectory.

    Returns:
        tuple:
            keep_running (bool): whether outer loop should continue
            has_data (bool): whether this episode produced any interactions
            episode_success (bool): whether this episode is successful
    """

    env.reset()
    env.render()

    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    print("\nHotkeys:")
    print("- Press F9  (default) to START recording")
    print("- Press F10 (default) to STOP current recording")
    print("- Press F12 (default) to EXIT script")

    # Keep track of previous gripper commands for non-active arms.
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    recording_active = False
    recorded_steps = 0

    while True:
        loop_start = time.time()

        if hotkeys.exit_requested:
            print("Exit hotkey received. Stopping collection loop.")
            return False, recorded_steps > 0, False

        if not recording_active:
            if hotkeys.consume_start():
                recording_active = True
                print("Recording started.")
            else:
                env.render()
                if max_fr is not None:
                    elapsed = time.time() - loop_start
                    diff = 1 / max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)
                continue

        if hotkeys.consume_stop():
            print("Recording stopped by hotkey.")
            break

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get newest action from input device
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # Device-side reset request (e.g. keyboard q)
        if input_ac_dict is None:
            print("Device requested reset. Ending current recording.")
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain previous gripper state for each robot and only update active robot.
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()
        recorded_steps += 1

        # Limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - loop_start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    has_data = recorded_steps > 0
    episode_success = bool(env._check_success()) if has_data else False
    print("Episode result: {}".format("SUCCESS" if episode_success else "FAILURE"))
    return True, has_data, episode_success


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into two hdf5 files:
    success and failure datasets.
    """
    def _write_split_hdf5(file_name, target_success):
        hdf5_path = os.path.join(out_dir, file_name)
        f = h5py.File(hdf5_path, "w")
        grp = f.create_group("data")

        num_eps = 0
        env_name = None

        for ep_directory in os.listdir(directory):
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            success = False

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or bool(dic["successful"])

            if len(states) == 0 or success != target_success:
                continue

            # The last state is one step ahead of the final action.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f_xml:
                xml_str = f_xml.read()
            ep_data_grp.attrs["model_file"] = xml_str
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))

        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
        grp.attrs["split"] = "success" if target_success else "failure"
        grp.attrs["num_demos"] = num_eps

        f.close()
        return hdf5_path, num_eps

    success_path, success_num = _write_split_hdf5("demo_success.hdf5", True)
    failure_path, failure_num = _write_split_hdf5("demo_failure.hdf5", False)
    print("Updated success dataset: {} ({} demos)".format(success_path, success_num))
    print("Updated failure dataset: {} ({} demos)".format(failure_path, failure_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Generic (e.g. BASIC) or config json path.",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simulation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense only) Reverse the effect of the x and y axes of the joystick.",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Goal update mode for device action generation.",
    )

    parser.add_argument(
        "--record-start-key",
        type=str,
        default="f9",
        help="Hotkey to start recording (single char or pynput Key name, e.g. f9)",
    )
    parser.add_argument(
        "--record-stop-key",
        type=str,
        default="f10",
        help="Hotkey to stop current recording (single char or pynput Key name)",
    )
    parser.add_argument(
        "--record-exit-key",
        type=str,
        default="f12",
        help="Hotkey to exit script (single char or pynput Key name)",
    )

    args = parser.parse_args()

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    env_info = json.dumps(config)

    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    elif args.device == "vr_osc":
        from robosuite.devices import VR_OSC

        device = VR_OSC(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard', 'spacemouse', 'dualsense', 'mjgui', or 'vr_osc'."
        )

    hotkeys = RecordingHotkeys(
        start_key=args.record_start_key,
        stop_key=args.record_stop_key,
        exit_key=args.record_exit_key,
    )

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    try:
        while True:
            keep_running, has_data, episode_success = collect_human_trajectory(
                env,
                device,
                args.arm,
                args.max_fr,
                args.goal_update_mode,
                hotkeys,
            )

            if has_data:
                env.set_successful(episode_success)
                env.flush_current_episode()
                gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
            else:
                print("No interaction was recorded in this episode. Skipping HDF5 update.")

            if not keep_running:
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting.")
    finally:
        hotkeys.close()
        env.close()
