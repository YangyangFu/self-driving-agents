import glob 
import os 
import json

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
def generate_recorder_info():
    recorder_info = []

    # read all replay log files
    log_dirs = ['ScenarioLogs']

    # read all subdirectories
    for log_dir in log_dirs:
        subdirs = glob.glob(os.path.join(FILE_PATH, f'{log_dir}/*'))
        for subdir in subdirs:
            info = {}
            info['folder'] = subdir.replace(FILE_PATH+'/', '')
            
            # find the *.log file
            log_file = glob.glob(f'{subdir}/*.log')
            info['name'] = log_file[0].split('/')[-1].split('.')[0]
            info['start_time'] = 0
            info['duration'] = 0
            recorder_info.append(info)

    # save the recorder info
    with open(f'{FILE_PATH}/recorder_info.json', 'w') as f:
        json.dump(recorder_info, f, indent=4)
        
    return recorder_info

        