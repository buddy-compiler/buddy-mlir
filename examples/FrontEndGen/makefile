#!/bin/bash
BUDDY_FRONTEND_GEN := ../../build/bin/buddy-frontendgen

frontendgen-emit-ast:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=ast

frontendgen-emit-antlr:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=antlr -g Toy

frontendgen-emit-visitor:
	@${BUDDY_FRONTEND_GEN} -f ./example.fegen -emit=visitor -g Toy

