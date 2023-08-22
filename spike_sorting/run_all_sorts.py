import os
import subprocess
from multiprocessing import Pool


# Define the function to run each script
def run_script(script):
    env = os.environ.copy()
    env["CONDA_DEFAULT_ENV"] = "spikesort"
    subprocess.call(["python", script], env=env)


# Define the main function
def main():
    # Set the directory where your scripts are located
    script_dir = "/home/kdriessen/gh_master/acr/spike_sorting/to_run"

    # Get a list of all the Python scripts in the directory
    scripts = [
        os.path.join(script_dir, f) for f in os.listdir(script_dir) if f.endswith(".py")
    ]

    for s in scripts:
        if "sort_utils" in s:
            scripts.remove(s)

    # Run the scripts in parallel using a Pool of workers
    with Pool() as p:
        p.map(run_script, scripts)


if __name__ == "__main__":
    main()
