from cs336_systems.utils import *
from cs336_systems.modal_utils import app, build_image, user_volume, secrets
import subprocess

@app.function(
    image=build_image(include_tests=True),
    timeout=2700,
    gpu="B200",
    volumes={"/root/data": user_volume},
)
def run_flash_tests():
    output = subprocess.run(
        ["uv", "run", "pytest", "/root/tests/", "-k", "sharded", "-v"],
        cwd="/root", capture_output=True, text=True,
    )
    print(output.stdout)
    print(output.stderr)
    return output.returncode
    


@app.local_entrypoint()
def run_tests():
    return_code = run_flash_tests.remote()
    assert return_code == 0, f"Tests failed with return code {return_code}"
    