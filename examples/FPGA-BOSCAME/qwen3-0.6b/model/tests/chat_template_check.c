/* Host harness: print the chat template the firmware will build, for comparison
 * against the official tokenizer's apply_chat_template. Records are
 * NUL-separated: <enable_thinking>\0<system>\0<user>\0 ... */
#include "tokenizer_resource.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  static unsigned char record[65536];
  static unsigned char out[65536];
  int c; size_t n = 0; int field = 0;
  char thinking[8] = {0}, system[8192] = {0}, user[8192] = {0};
  while ((c = fgetc(stdin)) != EOF) {
    if (c != '\0') { if (n + 1 < sizeof(record)) record[n++] = (unsigned char)c; continue; }
    record[n] = '\0';
    if (field == 0) { snprintf(thinking, sizeof(thinking), "%s", (char *)record); }
    else if (field == 1) { snprintf(system, sizeof(system), "%s", (char *)record); }
    else {
      snprintf(user, sizeof(user), "%s", (char *)record);
      size_t written = 0;
      int has_system = system[0] != '\0';
      if (qwen_chat_single_turn(out, sizeof(out) - 1, &written,
                                (const uint8_t *)system, has_system ? strlen(system) : 0,
                                has_system, (const uint8_t *)user, strlen(user),
                                thinking[0] == '1') != 0) {
        printf("ERROR\n");
      } else {
        out[written] = '\0';
        for (size_t i = 0; i < written; ++i) printf("%02x", out[i]);
        printf("\n");
      }
      field = -1;
    }
    ++field; n = 0;
  }
  return 0;
}
