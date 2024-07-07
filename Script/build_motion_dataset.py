import yaml
import argparse
import pickle
try:
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
except ImportError:
    import VclSimuBackend
    JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
from MoConVQCore.Utils.motion_dataset import HDF5MotionDataset
from MoConVQCore.Utils.misc import *


if __name__ == '__main__':
    '''
    convert mocap bvh into binary file, the bvh will be downsampled and some 
    important data(such as state and observation of each frame) 
    '''    
    parser = argparse.ArgumentParser()
    parser.add_argument("--using_yaml",  default=True, help="if true, configuration will be specified with a yaml file", action='store_true')
    parser.add_argument("--bvh_folder", type=str, default="", help="name of reference bvh folder")
    parser.add_argument("--env_fps", type=int, default=20, help="target FPS of downsampled reference motion")
    parser.add_argument("--env_scene_fname", type = str, default = "odecharacter_scene.pickle", help="pickle file for scene")
    parser.add_argument("--motion_dataset", type=str, default=None, help="name of output motion dataset")
    args = vars(parser.parse_args())
    
    if args['using_yaml']:
        config = load_yaml(initialdir='Data/Parameters/', path = r'Data/Parameters/bigdata.yml')
        args.update(config) 
        
    
    scene_loader = JsonSceneLoader()
    scene = scene_loader.file_load(args['env_scene_fname'])
    
    motion = HDF5MotionDataset(args['env_fps'])
    
    assert args['bvh_folder'] is not None
    motion.begin_create(args['motion_dataset'], args['clip_feature_set'])
    motion.add_folder_bvh(args['bvh_folder'], scene.character0, True)
    motion.end_create()