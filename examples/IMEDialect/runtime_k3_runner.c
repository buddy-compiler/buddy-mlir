// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Run the generated IME example on an A100 core before executing RVV/IME code.
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

extern int ime_example_main(void);

int main(void) {
  int fd = open("/proc/set_ai_thread", O_WRONLY);
  if (fd < 0) {
    perror("open /proc/set_ai_thread");
    return 1;
  }
  ssize_t written = write(fd, "0", 1);
  close(fd);
  if (written != 1) {
    perror("register AI thread");
    return 1;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(8, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask)) {
    perror("bind CPU 8");
    return 1;
  }
  unsigned long vlenb;
  __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
  if (vlenb != 128) {
    fprintf(stderr, "K3 A100 lowering requires VLEN=1024, got %lu\n",
            vlenb * 8);
    return 1;
  }
  return ime_example_main();
}
