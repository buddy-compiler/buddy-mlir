#!/bin/bash
BUDDY_FRONTEND_GEN := ../../../build/bin/buddy-frontendgen
BUDDY_FRONTEND_GEN_DSL = ../../../build/bin/buddy-frontendgen-dsl 

frontendgen-emit-ast:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=ast

frontendgen-emit-antlr:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=antlr -g Toy

frontendgen-emit-visiror:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=visitor -g Toy

frontendgendsl-constant-mlir:
	@${BUDDY_FRONTEND_GEN_DSL} ./constant.toy -emit=mlir 

frontendgendsl-constant-affine:
	@${BUDDY_FRONTEND_GEN_DSL} ./constant.toy -emit=mlir-affine

frontendgendsl-constant-jit:
	@${BUDDY_FRONTEND_GEN_DSL} ./constant.toy -emit=jit 

