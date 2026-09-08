import torch


def min_exllamav3_compute_capability():
    """
    Minimum compute capability required by the ExLlamaV3 kernels, per build.

    CUDA builds require Ampere (compute capability 8.0) or newer. ROCm builds
    use the ExLlamaV3 ROCm fork, which targets gfx1100-class GPUs (RDNA3,
    reported as compute capability 11.0 under ROCm PyTorch).
    """

    if torch.version.hip:
        return (11, 0)
    else:
        return (8, 0)


def hardware_supports_exllamav3(gpu_device_list: list[int]):
    """
    Check whether all GPUs in the list can run ExLlamaV3.

    On CUDA builds, ExLlamaV3 requires compute capability 8.0 (Ampere) or
    higher. On ROCm builds, the ExLlamaV3 ROCm fork requires gfx1100-class
    GPUs (RDNA3, compute capability 11.0) or higher.
    """

    min_required_capability = min_exllamav3_compute_capability()

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx) for device_idx in gpu_device_list
    )

    return min_compute_capability >= min_required_capability


def exllamav3_unsupported_gpu_message():
    """Error message for ExLlamaV3 GPU requirements, matching the torch build."""

    if torch.version.hip:
        return (
            "Unable to run ExllamaV3 because an unsupported GPU is "
            "found in this configuration. \n"
            "All GPUs must be RDNA3 "
            "(gfx1100-class, e.g. RX 7900 XTX) or newer under ROCm."
        )
    else:
        return (
            "Unable to run ExllamaV3 because an unsupported GPU is "
            "found in this configuration. \n"
            "All GPUs must be ampere "
            "(30 series) or newer. AMD GPUs are not supported."
        )
