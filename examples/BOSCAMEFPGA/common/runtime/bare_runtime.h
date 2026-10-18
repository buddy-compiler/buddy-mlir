//===- bare_runtime.h - Bare-metal C runtime interface --------------------===//
//
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
//
//===----------------------------------------------------------------------===//
//
// Lifecycle hooks for the NH bare-metal runtime on NR FPGA.
// _init() in bare_runtime.c calls them around main() in this order:
//   print_banner -> before_main -> main -> after_main.
// bare_runtime.c supplies weak no-op defaults; an application (e.g. hello.c)
// may provide strong definitions to customize banner / logging.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H

// Called from _init() in bare_runtime.c after UART init, before
// before_main()/main(). Override to print a boot banner.
void bare_runtime_print_banner(void);

// Called from _init() in bare_runtime.c immediately before main().
// Override to log entry into application code.
void bare_runtime_before_main(void);

// Called from _init() in bare_runtime.c after main() returns, before
// the final wfi park. Override to log completion.
void bare_runtime_after_main(void);

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
