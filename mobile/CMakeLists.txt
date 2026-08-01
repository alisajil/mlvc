cmake_minimum_required(VERSION 3.22)
project(mlvc_mobile CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QAIRT_INCLUDE points at the extracted QAIRT SDK include/QNN directory.
if(NOT DEFINED QAIRT_INCLUDE)
  set(QAIRT_INCLUDE "$ENV{HOME}/mlvc/qairt/qairt/2.45.0.260326/include/QNN")
endif()

add_executable(mlvc_bench
  src/main.cpp
  src/qnn_runner.cpp
)

add_executable(mlvc_encode
  src/mlvc_encode.cpp
  src/qnn_runner.cpp
  src/mlvc_codec.cpp
  src/rans.cpp
  src/srt_transport.cpp
)
# libsrt cross-compiled for arm64-android (see PLAN.md for the build recipe)
set(SRT_ANDROID "$ENV{HOME}/mlvc/srt/android-install")
target_include_directories(mlvc_encode PRIVATE ${SRT_ANDROID}/include)
target_link_libraries(mlvc_encode ${SRT_ANDROID}/lib/libsrt.a)

foreach(tgt mlvc_bench mlvc_encode)
  target_include_directories(${tgt} PRIVATE ${QAIRT_INCLUDE})
  target_link_libraries(${tgt} dl)
  target_compile_options(${tgt} PRIVATE -O2 -Wall)
endforeach()

# Camera app: NativeActivity shared lib packaged into an APK by build_apk.sh.
if(ANDROID)
  add_library(native_app_glue STATIC
    ${CMAKE_ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
  target_include_directories(native_app_glue PUBLIC
    ${CMAKE_ANDROID_NDK}/sources/android/native_app_glue)

  add_library(mlvccam SHARED
    src/camera_app.cpp
    src/qnn_runner.cpp
    src/mlvc_codec.cpp
    src/rans.cpp
  )
  target_include_directories(mlvccam PRIVATE ${QAIRT_INCLUDE})
  target_link_libraries(mlvccam native_app_glue android log camera2ndk mediandk dl)
  target_compile_options(mlvccam PRIVATE -O2 -Wall)
  # native_app_glue's ANativeActivity_onCreate must stay exported.
  set_target_properties(mlvccam PROPERTIES LINK_FLAGS "-u ANativeActivity_onCreate")
endif()
