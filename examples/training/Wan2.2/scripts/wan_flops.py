def estimate_flops(
    *,
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    sample_steps: int,
) -> float:
    num_parameters = 5_000_000_000  # TI2V-5B rough placeholder
    flops_per_forward = 6.0 * num_parameters
    total_flops = flops_per_forward * sample_steps * batch_size
    return float(total_flops)
