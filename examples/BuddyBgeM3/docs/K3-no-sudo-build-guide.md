# K3 板端免 sudo 构建手册(user@isrc)

> 取代 `K3上板操作手册.md` 的"路线 A"部分(该文档假设板上有 sudo / 能装包,已不适用)。
> 适用:`user@isrc`(Bianbu 4.0.1 / riscv64 / 16×X100 / 31G 内存 / 105G 空闲)
> 约束:**sudo 被拒 → 一切装到 `$HOME` 下,用户态**;WSL(x86_64)可与板子互传文件。

---

## 0. 现状与关键结论

**板上已有(实测)**:gcc/g++ 15.2、make 4.4.1、python3.14 + pip、git、网络(github 200)、16 核、31G 内存、105G 磁盘。
**板上缺**:`cmake`、`ninja`、任何 LLVM/buddy。
**结论**:只需要先在板上用户态补 `cmake` + `ninja`,然后就能原生编 LLVM24 + buddy-mlir。

**能复制什么、不能复制什么**:

| 类别 | 能否复制 | 说明 |
|---|---|---|
| WSL 的 cmake/ninja/llvm 二进制 | ❌ | x86_64 ELF,riscv64 上跑不了 |
| buddy-mlir **源码**(含 llvm submodule) | ✅ | 省板上 clone 大 submodule |
| **riscv64 版**工具包(cmake 预编译包等) | ✅ | 在 WSL 下载 riscv64 版本再传 |
| x86 生成的 **MLIR 图 + 权重**(`arg0.data` 等) | ✅ | 板上没 torch,靠它跳过 import |

---

## 1. 用户态安装 cmake + ninja(免 sudo)

### 1.1 装 cmake

**路线 A(快,若 WSL 能下到 riscv64 预编译包)**:在 WSL 里下载 → 传到板上解压:

```bash
# ===== 在 WSL(x86)上 =====
cd /tmp
curl -fLO https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-riscv64.tar.gz \
  || echo "该文件不存在 → 改用路线 B"
ls -lh cmake-3.31.6-linux-riscv64.tar.gz
# 传到板上(用户可直连):
scp cmake-3.31.6-linux-riscv64.tar.gz user@isrc:~/src/
```

```bash
# ===== 在 K3 板上 =====
mkdir -p ~/src ~/.local && cd ~/src
tar xf cmake-3.31.6-linux-riscv64.tar.gz
mv cmake-3.31.6-linux-riscv64/* ~/.local/
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc && export PATH=$HOME/.local/bin:$PATH
cmake --version | head -1
```

**路线 B(兜底,纯板上源码自举,约 10–20 分钟)**:WSL 下载不到预编译包时用。

```bash
# ===== 在 K3 板上 =====
sudo -n true 2>/dev/null || echo "(确认:无 sudo,走用户态)"
mkdir -p ~/src && cd ~/src
curl -fLO https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6.tar.gz
tar xf cmake-3.31.6.tar.gz && cd cmake-3.31.6
./bootstrap --prefix="$HOME/.local" --parallel="$(nproc)" -- \
    -DCMAKE_USE_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" && make install
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc && export PATH=$HOME/.local/bin:$PATH
cmake --version | head -1
```

> 若某个版本号 404,换成其它 `3.30.x` / `3.31.x`。

### 1.2 装 ninja(板上源码 self-bootstrap,只要 g++ + python3,1–2 分钟)

```bash
cd ~/src
curl -fLO https://github.com/ninja-build/ninja/archive/refs/tags/v1.12.1.tar.gz
tar xf v1.12.1.tar.gz && cd ninja-1.12.1
python3 configure.py --bootstrap        # 直接产出 ./ninja 二进制
mkdir -p ~/.local/bin && install -m755 ninja ~/.local/bin/ninja
ninja --version
```

> 若 `configure.py` 不存在(新版本移除),用已装好的 cmake 编:
> `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && install -m755 build/ninja ~/.local/bin/`
> 再不行就 `-G "Unix Makefiles"` 让后续 LLVM 构建直接用 make(见 §3 备注)。

### 1.3 自检

```bash
which cmake ninja; cmake --version | head -1; ninja --version
g++ --version | head -1; echo "GCC_DIR=$(dirname "$(g++ -print-libgcc-file-name)")"
nproc; free -g | head -2
```

---

## 2. 获取 buddy-mlir 源码(二选一)

**A. 从 WSL 复制(推荐,你已说可以传;省掉板上 clone 大 submodule)**

```bash
# ===== 在 WSL(x86)上 =====
cd /home/user/buddy-mlir
# 排除已构建产物(x86 的 build 很大且无用)
tar czf /tmp/buddy-src.tgz \
  --exclude='./build' --exclude='./llvm/build' --exclude='./.git' \
  --exclude='./models/*/build' .
ls -lh /tmp/buddy-src.tgz
scp /tmp/buddy-src.tgz user@isrc:~/buddy-k3/
```

```bash
# ===== 在 K3 板上 =====
cd ~/buddy-k3 && mkdir -p buddy-mlir && tar xzf buddy-src.tgz -C buddy-mlir
cd buddy-mlir && ls llvm/llvm/CMakeLists.txt && ls thirdparty
```

> ⚠️ 不含 `.git` 就**不能** `git submodule` 操作,但源码树已完整(llvm 源码在 `llvm/llvm`),足够构建。以后要提交 patch 再单独 `git clone`。

**B. 板上直接 clone(网络通,也可)**

```bash
cd ~/buddy-k3
git clone --recurse-submodules --shallow-submodules \
  https://github.com/buddy-compiler/buddy-mlir.git
cd buddy-mlir && git submodule status llvm | head -1
```

> `llvm` submodule 是 **RuyiAI-Stack/llvm-project 的 riscv 分支**(commit `2d26d272a`,带 BOSC AME/IME 后端),不要换成上游 llvm。

---

## 3. 原生编译 LLVM 24(rv64,约 1–2 小时)

```bash
cd ~/buddy-k3/buddy-mlir
GCC_DIR="$(dirname "$(g++ -print-libgcc-file-name)")"   # Bianbu 上一般是 /usr/lib/gcc/riscv64-linux-gnu/15

cmake -G Ninja -S llvm/llvm -B llvm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBXML2=OFF \
  "-DRUNTIMES_CMAKE_ARGS=-DCMAKE_C_FLAGS=--gcc-install-dir=${GCC_DIR};-DCMAKE_CXX_FLAGS=--gcc-install-dir=${GCC_DIR};-DCMAKE_HAVE_LIBC_PTHREAD=TRUE;-DCMAKE_USE_PTHREADS_INIT=TRUE;-DCMAKE_THREAD_LIBS_INIT=" \
  -DPython3_EXECUTABLE="$(which python3)"

ninja -C llvm/build -j"$(nproc)"
```

> - 关掉了 python bindings / zlib / zstd / terminfo 等**需要有 dev 头但板子没 sudo 装**的可选项;这些对"编 rax + 跑"都不需要。
> - `--gcc-install-dir` 是 RISC-V 原生平台上让 clang 找到 gcc 运行时(crt/libgcc)的官方做法(见 `.github/actions/ci-common/action.yml` 的 riscv64 分支)。
> - 若 ninja 最终没装上:把 `-G Ninja` 换成 `-G "Unix Makefiles"`,`ninja -C llvm/build` 换成 `make -C llvm/build -j"$(nproc)"`。
> - 建议挂后台:`nohup ninja -C llvm/build -j"$(nproc)" > ~/llvm-build.log 2>&1 &` 然后 `tail -f ~/llvm-build.log`。

**成功标志**:

```bash
ls llvm/build/bin/{llc,mlir-opt,mlir-translate,clang,clang++} | head
llvm/build/bin/llc --version | grep -i riscv          # 应有 riscv64
ls llvm/build/bin/../lib/libomp.so 2>/dev/null || find llvm/build -name 'libomp.so' | head
find llvm/build -name 'libmlir_c_runner_utils.so*' | head   # rax 运行时需要
```

---

## 4. 原生编译 buddy-mlir

```bash
cd ~/buddy-k3/buddy-mlir
cmake -G Ninja -S . -B build \
  -DMLIR_DIR="$PWD/llvm/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=OFF \
  -DPython3_EXECUTABLE="$(which python3)"

ninja -C build -j"$(nproc)"
```

**冒烟验证**:

```bash
ls build/bin/{buddy-opt,buddy-cli,buddy-server,buddy-translate}
build/bin/buddy-opt --version | head -1
build/bin/buddy-cli --help | head -5
```

> 板上**不需要** torch:MLIR 图与权重沿用 WSL 端生成的,rax 只在板上做"编译 + 链接 + 打包"(见 §5)。

---
---

## 6. 常见坑

- **`sudo: I'm sorry`** → 你无提权;本文档全程不依赖 sudo。
- **复制 x86 的 cmake/ninja 过去报错的** → 架构不符,重新下 riscv64 版或板上自举。
- **cmake bootstrap 报 OpenSSL 找不到** → 已加 `-DCMAKE_USE_OPENSSL=OFF`。
- **LLVM 编到一半 OOM** → 降到 `-j8`(31G 内存通常够,但 clang 编大文件吃内存)。
- **python3.14 太新导致某脚本报错** → 构建 LLVM/buddy 只把它当 interpreter;真出问题再考虑用户态装 python3.12(conda/uv)。
- **板上没有 torch / transformers** → 正常,按 §5 复用 WSL 的 MLIR + 权重,不需要板上装 torch。
