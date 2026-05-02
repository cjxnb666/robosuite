import os
import glob
import subprocess
import h5py

input_dir = '/Users/chuangjianxue/Desktop/coding/robosuite-master/datasets/nut_assembly_mjgui'
output_dir = '/Users/chuangjianxue/Desktop/coding/robosuite-master/datasets/act_training_data'
convert_script = '/Users/chuangjianxue/Desktop/coding/robosuite-master/robosuite/scripts/convert_to_act.py'

os.makedirs(output_dir, exist_ok=True)

hdf5_files = glob.glob(os.path.join(input_dir, '*', 'demo.hdf5'))
# 兼容刚刚改的新代码（会生成 demo_success.hdf5）
hdf5_files.extend(glob.glob(os.path.join(input_dir, '*', 'demo_success.hdf5')))

print(f"Found {len(hdf5_files)} HDF5 files to check in {input_dir}.")

valid_files = []
for fpath in hdf5_files:
    try:
        with h5py.File(fpath, 'r') as f:
            if 'data' in f and len(list(f['data'].keys())) > 0:
                valid_files.append(fpath)
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

print(f"Found {len(valid_files)} valid HDF5 files with demonstrations.")

global_ep_idx = 0

for fpath in valid_files:
    folder_name = os.path.basename(os.path.dirname(fpath))
    tmp_out = os.path.join(output_dir, f"tmp_{folder_name}")
    print(f"\n--- Converting {fpath} to {tmp_out} ---")
    
    cmd = [
        "/Users/chuangjianxue/Desktop/coding/robosuite-master/robosuite_env/bin/python",
        convert_script,
        "--input", fpath,
        "--output", tmp_out
    ]
    subprocess.run(cmd, check=True)
    
    # 移动出来并重命名为全局唯一的 episode_N.hdf5
    generated_eps = glob.glob(os.path.join(tmp_out, 'episode_*.hdf5'))
    for ep in generated_eps:
        new_name = os.path.join(output_dir, f'episode_{global_ep_idx}.hdf5')
        os.rename(ep, new_name)
        global_ep_idx += 1
    
    # 删除临时目录
    if os.path.exists(tmp_out):
        os.rmdir(tmp_out)

print(f"\nAll done! Successfully generated {global_ep_idx} ACT-formatted episodes in {output_dir}")
