# Get all `.py` files in the current path
file(GLOB PY_FILES "*.py")

# Copy all `.py` files to the destination
file(COPY ${PY_FILES} DESTINATION ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/)
