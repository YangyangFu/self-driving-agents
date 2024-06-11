"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
# Usage: Collect Data or Eval Agent
# Message: All references pls check the readme
"""
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
from experts.utils.utils import bcolors as bc
from experts.utils.utils import CarlaServerManager
import hydra
from hydra.utils import get_original_cwd, to_absolute_path

import os, sys
import time
import gc
import logging

log = logging.getLogger(__name__)

"""
TODO: redesign the configuration files
@dataclass
class BaseSensorConfig:
    pass 
class CameraConfig(BaseSensorConfig):
    pass 
class LidarConfig(BaseSensorConfig):
    pass 

@dataclass
class BaseAgentConfig:
    pass 

class AutoAgentConfig(BaseAgentConfig):
    pass
"""

@hydra.main(config_path="config", config_name="experiment")
def main(args):
    # config init
    args.routes    = os.path.join(args.absolute_path, args.routes)
    args.agent     = os.path.join(args.absolute_path, args.agent)
    args.checkpoint = os.path.join(args.absolute_path, args.checkpoint)
    args.debug_checkpoint = os.path.join(args.absolute_path, args.debug_checkpoint)
    
    # for multi carla
    args.traffic_manager_port = args.port + 6000

    # shared paremeters between experiment and agent
    args.agent_config.routes = args.routes
    args.agent_config.routes_subset = args.routes_subset
    
    # ======= agent specific parameters
    args.agent_config.output_dir = os.path.join(args.absolute_path, args.agent_config.output_dir)
    
    # for autopilot cache:
    if hasattr(args.agent_config, 'birdview_cache_dir'):
        args.agent_config.birdview_cache_dir = to_absolute_path(args.agent_config.birdview_cache_dir)
    
    # start CARLA
    print('-'*20 + "TEST Agent: " + bc.OKGREEN + args.agent.split('/')[-1] + bc.ENDC + '-'*20)

    # run official leaderboard ====>
    statistics_manager = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)
    crashed = leaderboard_evaluator.run(args)
    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    
    start_time = time.time()
    main()
    print('clean memory on no.', gc.collect(), "Uncollectable garbage:", gc.garbage)
    print(f"{bc.OKGREEN}TOTAL RUNNING TIME{bc.ENDC}: --- %s hours ---" % round((time.time() - start_time)/3600,2))