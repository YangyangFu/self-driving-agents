import glob 

def generate_recorder_info():
    recorder_info = []

    # read all replay log files
    log_dirs = ['RouteLogs', 'ScenarioLogs']

    # read all subdirectories
    for log_dir in log_dirs:
        subdirs = glob.glob(f'{log_dir}/*')
        for subdir in subdirs:
            info = {}
            info['folder'] = subdir
            
            # find the *.log file
            log_file = glob.glob(f'{subdir}/*.log')
            info['name'] = log_file[0].split('/')[-1].split('.')[0]
            info['start_time'] = 0
            info['end_time'] = 0
            recorder_info.append(info)

    return recorder_info

        