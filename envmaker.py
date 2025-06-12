import yaml
import subprocess
import logging

# Configure logging
def setup_logging(log_file="env_setup.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_cmd(cmd, *, check=True):
    logging.info(f"Running command: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        if result.stdout:
            logging.info(result.stdout.strip())
        if result.stderr:
            logging.warning(result.stderr.strip())
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed (code={e.returncode}): {e.stderr.strip()}")
        if check:
            raise
        return e.returncode


def main():
    # Load environment YAML
    with open('environment.yml', 'r') as f:
        env_data = yaml.safe_load(f)

    env_name = env_data.get('name', 'new_env')
    deps = env_data.get('dependencies', [])
    channels = env_data.get('channels', [])

    # Build channel flags for conda commands
    channel_flags = ' '.join(f"-c {ch}" for ch in channels)

    # Determine Python version
    python_version = next(
        (dep.split('=')[1] for dep in deps if isinstance(dep, str) and dep.startswith('python=')),
        '3.8'
    )

    # Step 1: create base environment with channels
    run_cmd(f"conda create --name {env_name} python={python_version} {channel_flags} --yes")

    # Step 2: install conda packages (no dependencies) with channels
    for dep in deps:
        if isinstance(dep, str) and not dep.startswith(('python=', 'pip')):
            run_cmd(f"conda install --name {env_name} {dep} {channel_flags} --no-deps --yes")

    # Step 2.5: ensure CMake is available for building packages
    run_cmd(f"conda install --name {env_name} cmake {channel_flags} --yes")

    # Step 3: install pip packages inside conda env, preserving flags
    for dep in deps:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_list = dep['pip']
            # Separate pip flags (e.g. --extra-index-url) from actual packages
            pip_flags = [item for item in pip_list if item.startswith('-')]
            pip_pkgs = [item for item in pip_list if not item.startswith('-')]
            # Install all pip packages via pip (including dlib)
            if pip_pkgs:
                flags_str = ' '.join(pip_flags)
                pkgs_str = ' '.join(pip_pkgs)
                run_cmd(f"conda run --name {env_name} pip install {flags_str} {pkgs_str}")


if __name__ == '__main__':
    setup_logging()
    try:
        main()
        logging.info("Environment setup completed successfully.")
    except Exception as e:
        logging.critical(f"Setup terminated due to error: {e}")
        exit(1)

