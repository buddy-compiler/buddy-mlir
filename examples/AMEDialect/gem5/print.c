// Print one signed i32 matrix row without a target C library.
// The host fallback lets the formatter and short-write handling run natively.

_Static_assert(sizeof(int) == 4, "print_C requires 32-bit integers");

#if defined(__riscv) && __riscv_xlen == 64
static long write_stdout(const char *data, unsigned long size) {
  register long fd __asm__("a0") = 1;
  register const char *buffer __asm__("a1") = data;
  register unsigned long count __asm__("a2") = size;
  register long number __asm__("a7") = 64;
  __asm__ volatile("ecall"
                   : "+r"(fd)
                   : "r"(buffer), "r"(count), "r"(number)
                   : "memory");
  return fd;
}
#else
#include <errno.h>
#include <unistd.h>

static long write_stdout(const char *data, unsigned long size) {
  long written = write(STDOUT_FILENO, data, size);
  return written < 0 ? -errno : written;
}
#endif

static char *append_i32(char *output, int value) {
  char reversed[10];
  unsigned int digits = 0;
  unsigned int magnitude = (unsigned int)value;
  if (value < 0) {
    *output++ = '-';
    // Unsigned subtraction also handles the magnitude of INT_MIN.
    magnitude = 0u - magnitude;
  }
  do {
    reversed[digits++] = (char)('0' + magnitude % 10);
    magnitude /= 10;
  } while (magnitude);
  while (digits)
    *output++ = reversed[--digits];
  return output;
}

// Return zero only after the complete row has reached standard output.
int print_C(int v0, int v1, int v2, int v3) {
  char line[48]; // Four INT_MIN strings, three spaces and one newline.
  char *end = append_i32(line, v0);
  *end++ = ' ';
  end = append_i32(end, v1);
  *end++ = ' ';
  end = append_i32(end, v2);
  *end++ = ' ';
  end = append_i32(end, v3);
  *end++ = '\n';

  const char *next = line;
  unsigned long remaining = (unsigned long)(end - line);
  while (remaining) {
    long written = write_stdout(next, remaining);
    if (written == -4) // Linux EINTR: retry without consuming output.
      continue;
    if (written <= 0 || (unsigned long)written > remaining)
      return 1;
    next += written;
    remaining -= (unsigned long)written;
  }
  return 0;
}
