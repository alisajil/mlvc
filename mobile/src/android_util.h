#pragma once
// Small JNI/asset helpers shared by camera_app.cpp (encode/sender) and
// decode_loop.cpp (receiver) - NativeActivity has no native accessor for
// intent extras or app info, so these go through JNI reflection once at
// startup.

#include <android_native_app_glue.h>

#include <string>

namespace mlvc {

// Reads a string extra from the launching intent (empty if absent).
std::string intentExtra(android_app* app, JNIEnv* env, const char* name);

// This app's own native library directory (jniLibs path inside the APK) -
// fastrpc/ADSP_LIBRARY_PATH needs this to find the Hexagon skel.
std::string nativeLibDir(android_app* app, JNIEnv* env);

// Copies an APK asset to the app's private files dir (skipping the copy if
// a same-size file is already there) and returns the destination path -
// QnnRunner needs a real filesystem path, not an APK-internal one.
std::string materializeAsset(android_app* app, const char* name);

}  // namespace mlvc
