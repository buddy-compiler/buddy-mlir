# YOLOv8 object detection / pose estimation model integration.
#
# YOLOv8 uses a single forward graph, so it keeps custom import and manifest
# scripts while reusing buddy_add_model for runner/plugin/.rax wiring.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_YOLOV8_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/k3_pose_320.json"
  CACHE FILEPATH "YOLOv8 variant spec JSON")
set(BUDDY_YOLOV8_CHECKPOINT ""
  CACHE FILEPATH "Ultralytics YOLOv8 checkpoint (empty: Ultralytics downloads the spec's checkpoint)")

buddy_add_model(
  NAME yolov8
  MODEL_KIND single_forward
  SPEC "${BUDDY_YOLOV8_SPEC}"
  RUNNER_SRC YoloV8Runner.cpp
  RUNNER_PLUGIN_SRC YoloV8RunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_YOLOV8_CHECKPOINT}"
  LOCAL_MODEL_ENV YOLOV8_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-yolov8.py"
  LOWER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_yolov8_manifest.py"
  MODEL_SO_NAME yolov8_model.so
  ASSET_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/labels.txt"
)
