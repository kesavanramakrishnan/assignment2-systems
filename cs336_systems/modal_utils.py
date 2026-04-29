from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "kesavanr"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"systems-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"basics-{SUNET_ID}", create_if_missing=True, version=2)



def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim()
        .apt_install("wget", "gzip")
        .run_commands(
            "wget -nv https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
        .add_local_dir(
            "cs336-basics",
            remote_path="/.uv/cs336-basics",
            copy=True,
        )
        .uv_sync()
        .add_local_python_source("cs336_systems")
    )
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"wandb_secret": "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"})]
    return secrets



