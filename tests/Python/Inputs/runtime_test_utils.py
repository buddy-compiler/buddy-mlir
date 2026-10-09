import platform


def llc_target_args():
    """Match the native Linux C compiler's floating-point ABI on RISC-V."""
    if platform.machine() == "riscv64":
        return [
            "-mtriple=riscv64-unknown-linux-gnu",
            "-target-abi=lp64d",
            "-mattr=+f,+d",
        ]
    return []
