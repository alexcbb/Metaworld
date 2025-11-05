import gymnasium as gym
import metaworld
import matplotlib.pyplot as plt
from metaworld import MT1
import random

from argparse import ArgumentParser
from metaworld.asset_path_utils import full_V3_path_for


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--generalization', type=str, default='', help='generalization to test')
    args = parser.parse_args()

    generalization = args.generalization
    task_name = 'assembly-v3'
    print(f"Testing {task_name} with generalization: {generalization}")
    if generalization is not '':
        assert(generalization in ['distractor_easy', 'distractor_medium', 'distractor_hard', 'blue-woodtable', \
                                    'dark-woodtable', 'darkwoodtable', 'cast_right', 'cast_left', 'darker', 'brighter'], \
                                    "generalization should be one of the existing generalizations")
        if 'assembly' in task_name:
            model_path = full_V3_path_for(f"sawyer_xyz/sawyer_assembly_peg_{generalization}.xml")
        elif 'bin' in task_name:
            model_path = full_V3_path_for(f"sawyer_xyz/sawyer_bin_picking_{generalization}.xml")
        elif 'button' in task_name:
            model_path = full_V3_path_for(f"sawyer_xyz/sawyer_button_press_{generalization}.xml")
        elif 'drawer' in task_name:
            model_path = full_V3_path_for(f"sawyer_xyz/sawyer_drawer_{generalization}.xml")
        elif 'hammer' in task_name:
            model_path = full_V3_path_for(f"sawyer_xyz/sawyer_hammer_{generalization}.xml")
    else:
        model_path = full_V3_path_for(f'sawyer_xyz/sawyer_{task_name.replace("-", "_").replace("_v3", "")}.xml')

    seed = 42 # for reproducibility
    mt1 = MT1(task_name, seed=seed)
    task = random.choice(mt1.train_tasks)
    env = mt1.train_classes[task_name](
        render_mode="rgb_array", 
        camera_name="corner2",
        model_name=model_path,
    )
    env.set_task(task)
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
    obs, info = env.reset()

    a = env.action_space.sample() # randomly sample an action
    obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
    plt.imshow(env.render()[::-1])
    plt.axis('off')
    plt.show()
