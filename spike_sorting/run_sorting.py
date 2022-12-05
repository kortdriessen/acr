import os

dir = '/home/kdriessen/github_t2/acr/spike_sorting/to_run/'
files_to_run = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
for file in files_to_run:
    if file == 'sort_utils.py':
        continue
    print(f'running {file}')
    os.system(f'python {dir}{file}')