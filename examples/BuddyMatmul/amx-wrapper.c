#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define ARCH_REQ_XCOMP_PERM 0x1023

extern void _mlir_ciface_amx_main();

// #define ARCH_REQ_XCOMP_PERM 0x1023

// External functions
// extern void _mlir_ciface_amx_main();
// extern long long get_time_us();
// extern void print_timing(long long start_us, long long end_us);

// MLIR rtclock function implementation
double _mlir_ciface_rtclock() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("Checking AMX support...\n");

    // Try to request permission to use AMX
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18); // 18 = AMX_TILE
    if (ret != 0) {
        printf("Warning: Failed to request AMX_TILE permission: %ld (errno: %s)\n", ret, strerror(errno));
        printf("This might be due to kernel version or configuration.\n");
        printf("Attempting to run anyway...\n");
    } else {
        printf("AMX_TILE permission granted\n");
    }

    ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 19); // 19 = AMX_DATA
    if (ret != 0) {
        printf("Warning: Failed to request AMX_DATA permission: %ld (errno: %s)\n", ret, strerror(errno));
        printf("Attempting to run anyway...\n");
    } else {
        printf("AMX_DATA permission granted\n");
    }

    printf("Starting AMX computation...\n");

    // Call the MLIR-generated main function
    _mlir_ciface_amx_main();



    printf("AMX computation completed successfully!\n");

    return 0;
}
