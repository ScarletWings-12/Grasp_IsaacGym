#External Libraries
import numpy as np
import hydra
from tqdm import tqdm
import os
import argparse
import sys
import time

def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Standalone script for grasp filtering.')
    parser.add_argument('--headless', type=bool, help='Running Program in headless mode',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--force_reset', type=bool, help='Force Reset of Isaac Sim',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--json_dir', type=str, help='Directory of Grasp Information', default='')
    parser.add_argument('--gripper_dir', type=str, help='Directory of Gripper urdf/usd', default='')
    parser.add_argument('--objects_dir', type=str, help='Directory of Object usd', default='')
    parser.add_argument('--output_dir', type=str, help='Output directroy for filterd grasps', default='')
    parser.add_argument('--num_w', type=int, help='Number of Workstations used in the simulation', default=150)
    parser.add_argument('--device', type=int, help='Gpu to use', default=0)
    parser.add_argument('--test_time', type=int, help='Total time for each grasp test', default=3)
    parser.add_argument('--print_results', type=bool, help='Enable printing of grasp statistics after filtering a document',
                         default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--controller', type=str,
                        help='Gripper Controller to use while testing, should match the controller dictionary in the Manager Class',
                        default='default')
    parser.add_argument('--/log/level', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/fileLogLevel', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/outputStreamLevel', type=str, help='isaac sim logging arguments', default='', required=False)
    
    return parser

#Parser
parser = make_parser()
args = parser.parse_args()
head = args.headless
force_reset = args.force_reset
print(args.controller)

#launch Isaac Sim before any other imports
from omni.isaac.kit import SimulationApp
config= {
    "headless": head,
    'max_bounces':0,
    'fast_shutdown': True,
    'max_specular_transmission_bounces':0,
    'physics_gpu': args.device,
    'active_gpu': args.device
    }
simulation_app = SimulationApp(config) # we can also run as headless.


#World Imports
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner    # import Cloner interface
from omni.isaac.core.utils.stage import add_reference_to_stage

# Custom Classes
from manager import Manager
from views import View

#Omni Libraries
from omni.isaac.core.utils.stage import add_reference_to_stage,open_stage, save_stage
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_children, get_prim_path, get_prim_at_path
from omni.isaac.core.utils.transformations import pose_from_tf_matrix


def import_gripper(work_path,usd_path, EF_axis):
        """ Imports Gripper to World

        Args:
            work_path: prim_path of workstation
            usd_path: path to .usd file of gripper
            EF_axis: End effector axis needed for proper positioning of gripper
        
        """
        T_EF = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
        if (EF_axis == 1):
            T_EF = np.array([[ 0,0,1,0],
                            [ 0,1,0,0],
                            [-1,0,0,0],
                            [0,0,0,1]])
        elif (EF_axis == 2):
            T_EF = np.array([[1, 0,0,0],
                                [0, 0,1,0],
                                [0,-1,0,0],
                                [0, 0,0,1]])
        elif (EF_axis == 3):
            T_EF = np.array([[1, 0, 0,0],
                                [0,-1, 0,0],
                                [0, 0,-1,0],
                                [0, 0, 0,1]])
        elif (EF_axis == -1):
            T_EF = np.array([[0,0,-1,0],
                                [0,1, 0,0],
                                [1,0, 0,0],
                                [0,0, 0,1]])
        elif (EF_axis == -2):
            T_EF = np.array([[1,0, 0,0],
                                [0,0,-1,0],
                                [0,1, 0,0],
                                [0,0, 0,1]])
        #Robot Pose
        gripper_pose= pose_from_tf_matrix(T_EF.astype(float))
        
        # Adding Robot usd
        add_reference_to_stage(usd_path=usd_path, prim_path=work_path+"/gripper")
        robot = world.scene.add(Articulation(prim_path = work_path+"/gripper", name="gripper",
                            position = gripper_pose[0], orientation = gripper_pose[1], enable_dof_force_sensors = True))
        robot.set_enabled_self_collisions(False)
        return robot, T_EF

def import_object(work_path, usd_path):
    """ Import Object .usd to World

    Args:
        work_path: prim_path to workstation
        usd_path: path to .usd file of object
    """
    add_reference_to_stage(usd_path=usd_path, prim_path=work_path+"/object")
    object_parent = world.scene.add(GeometryPrim(prim_path = work_path+"/object", name="object"))
    l = get_prim_children(object_parent.prim)
    #print(l)

    prim = get_prim_at_path(work_path+"/object"+ '/base_link/collisions/mesh_0')
    '''
    MassAPI = UsdPhysics.MassAPI.Get(world.stage, prim.GetPath())
    try: 
        og_mass = MassAPI.GetMassAttr().Get()
        if og_mass ==0:
            og_mass = 1
            print("Failure reading object mass, setting to default value of 1 kg.")
    except:
        og_mass = 1
        print("Failure reading object mass, setting to default value of 1 kg.")

    # Create Rigid Body attribute
    og_mass = 1
    '''

    object_prim = RigidPrim(prim_path= get_prim_path(l[0]))

    mass= 1 #Deprecated use of mass for gravity 

    return object_parent, mass


if __name__ == "__main__":
    
    # ./python.sh /home/szh/Documents/isaac_sim_grasping/standalone.py --output_dir=/home/szh/Documents/Outputs --gripper_dir=/home/szh/Documents/isaac_sim_grasping/grippers --objects_dir=/home/szh/Documents/GoogleScannedObjects_US
    
    
    # Directories
    grippers_directory = "/home/szh/Documents/Grasp_Isaacgym/grippers"
    objects_directory = "/home/szh/Documents/Grasp_Isaacgym/grippers"
    output_directory = "/home/szh/Documents/Grasp_Isaacgym/output"
    # grippers_directory = args.gripper_dir
    # objects_directory = args.objects_dir
    # output_directory = args.output_dir
    
    if not os.path.exists(grippers_directory):
        raise ValueError("Grippers directory not given correctly")
    elif not os.path.exists(objects_directory):
        raise ValueError("Objects directory not given correctly")
    elif not os.path.exists(output_directory): 
        raise ValueError("Output directory not given correctly")

    # Testing Hyperparameters
    num_w = args.num_w
    test_time = args.test_time
    verbose = args.print_results
    controller = args.controller
    #physics_dt = 1/120

    world = World(set_defaults = False)