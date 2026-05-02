import h5py
import cv2
import argparse
import numpy as np

def visualize_act_hdf5(filepath, fps=30):
    print(f"Loading {filepath}...")
    try:
        with h5py.File(filepath, 'r') as f:
            agentview = f['observations/images/agentview'][()]
            eye_in_hand = f['observations/images/robot0_eye_in_hand'][()]
            actions = f['action'][()]
            
            num_frames = agentview.shape[0]
            print(f"Total frames: {num_frames}")
            print(f"Image shape: {agentview.shape[1:]}")
            print(f"Action dim: {actions.shape[1]}")
            
            delay = int(1000 / fps)
            
            print("\nControls:")
            print("- Press 'SPACE' to pause/play")
            print("- Press 'q' or 'ESC' to quit")
            print("- When paused, press 'd' for next frame, 'a' for previous frame")
            
            paused = False
            i = 0
            while i < num_frames:
                # MuJoCo 渲染出来的是 RGB 格式，而 OpenCV 默认显示是 BGR 格式
                # 所以我们需要转换一下颜色通道才能颜色正常
                img_agent_bgr = cv2.cvtColor(agentview[i], cv2.COLOR_RGB2BGR)
                img_wrist_bgr = cv2.cvtColor(eye_in_hand[i], cv2.COLOR_RGB2BGR)
                
                # 把两个画面左右拼接到一起
                combined_img = np.hstack((img_agent_bgr, img_wrist_bgr))
                
                # 在画面上打上帧号文字
                cv2.putText(combined_img, f"Frame: {i}/{num_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined_img, "Agent View", (10, combined_img.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined_img, "Wrist View", (img_agent_bgr.shape[1] + 10, combined_img.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("ACT Dataset Viewer", combined_img)
                
                key = cv2.waitKey(0 if paused else delay) & 0xFF
                
                if key == ord('q') or key == 27: # ESC
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('d') and paused:
                    i = min(i + 1, num_frames - 1)
                elif key == ord('a') and paused:
                    i = max(i - 1, 0)
                else:
                    if not paused:
                        i += 1
                        
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error reading HDF5: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to episode_X.hdf5')
    parser.add_argument('--fps', type=int, default=30, help='Playback framerate')
    args = parser.parse_args()
    
    visualize_act_hdf5(args.file, args.fps)
