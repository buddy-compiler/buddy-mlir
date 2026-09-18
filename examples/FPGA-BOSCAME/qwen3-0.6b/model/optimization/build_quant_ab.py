#!/usr/bin/env python3
"""Compare production scalar/RVV quantize kernels with identical NR runtime."""
from build_dequant_ab import main

if __name__ == "__main__":
    main(benchmark="quant")
