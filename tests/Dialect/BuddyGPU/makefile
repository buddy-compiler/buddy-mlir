# 定义路径
BUDDY_OPT = /home/wuxintong/buddy-mlir/build/bin/buddy-opt
FILECHECK = /home/wuxintong/buddy-mlir/llvm/build/bin/FileCheck
TEST_FILE = /home/wuxintong/buddy-mlir/tests/Dialect/BuddyGPU/hoist-static-alloc.mlir

# 默认目标
all: test

# 测试命令
test:
	$(BUDDY_OPT) --split-input-file --transform-interpreter $(TEST_FILE) | $(FILECHECK) $(TEST_FILE)

# 清理命令（如果需要）
clean:
	@echo "Nothing to clean for this test."
