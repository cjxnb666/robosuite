import os
import json
import h5py
import numpy as np
import argparse
import robosuite as suite

def convert_dataset(input_path, out_dir, max_steps=None):
    os.makedirs(out_dir, exist_ok=True)
    f_in = h5py.File(input_path, 'r')
    
    env_info = json.loads(f_in['data'].attrs['env_info'])
    # 覆盖渲染设置，强制开启离屏渲染以获取相机图像
    env_info['has_renderer'] = False
    env_info['has_offscreen_renderer'] = True
    env_info['use_camera_obs'] = True
    env_info['camera_names'] = ['agentview', 'robot0_eye_in_hand']
    # ACT 常用分辨率
    env_info['camera_heights'] = 240
    env_info['camera_widths'] = 320
    
    print("Initializing environment for rendering...")
    env = suite.make(**env_info)
    
    demos = list(f_in['data'].keys())
    print(f"Found {len(demos)} demonstrations.")
    
    for ep_idx, ep in enumerate(demos):
        print(f"\nProcessing {ep}...")
        states = f_in[f'data/{ep}/states'][()]
        actions = f_in[f'data/{ep}/actions'][()]
        model_xml = f_in[f'data/{ep}'].attrs['model_file']
        
        if max_steps is not None:
            states = states[:max_steps]
            actions = actions[:max_steps]
            
        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        
        images_agent = []
        images_wrist = []
        qpos_list = []
        
        for t, state in enumerate(states):
            if t % 20 == 0:
                print(f"  Rendering Step {t}/{len(states)}")
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obs = env._get_observations()
            
            # ACT 要求的图像数据通常为 uint8 [H, W, 3]
            img_agent = obs['agentview_image']
            img_wrist = obs['robot0_eye_in_hand_image']
            
            # ACT 要求的 proprioception (关节角度 qpos)，包含 7个手臂关节 + 2个夹爪关节
            qpos = np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']])
            
            images_agent.append(img_agent)
            images_wrist.append(img_wrist)
            qpos_list.append(qpos)
            
        ep_file = os.path.join(out_dir, f'episode_{ep_idx}.hdf5')
        print(f"Saving to {ep_file}...")
        with h5py.File(ep_file, 'w') as f_out:
            f_out.create_dataset('action', data=actions)
            obs_grp = f_out.create_group('observations')
            obs_grp.create_dataset('qpos', data=np.array(qpos_list, dtype=np.float32))
            img_grp = obs_grp.create_group('images')
            img_grp.create_dataset('agentview', data=np.array(images_agent, dtype=np.uint8))
            img_grp.create_dataset('robot0_eye_in_hand', data=np.array(images_wrist, dtype=np.uint8))
            
        print(f"Done saving {ep_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to raw demo.hdf5')
    parser.add_argument('--output', type=str, required=True, help='Dir to save ACT formatted episodes')
    parser.add_argument('--max_steps', type=int, default=None, help='Limit steps for testing')
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output, args.max_steps)
