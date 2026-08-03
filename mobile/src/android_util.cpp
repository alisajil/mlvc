#include "android_util.h"

#include <android/asset_manager.h>
#include <android/log.h>

#include <cstdio>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MLVC", __VA_ARGS__)

namespace mlvc {

std::string intentExtra(android_app* app, JNIEnv* env, const char* name) {
  jobject activity = app->activity->clazz;
  jclass actCls = env->GetObjectClass(activity);
  jmethodID getIntent = env->GetMethodID(actCls, "getIntent", "()Landroid/content/Intent;");
  jobject intent = env->CallObjectMethod(activity, getIntent);
  if (!intent) return "";
  jclass intCls = env->GetObjectClass(intent);
  jmethodID getStr =
      env->GetMethodID(intCls, "getStringExtra", "(Ljava/lang/String;)Ljava/lang/String;");
  jstring key = env->NewStringUTF(name);
  auto val = static_cast<jstring>(env->CallObjectMethod(intent, getStr, key));
  if (!val) return "";
  const char* c = env->GetStringUTFChars(val, nullptr);
  std::string s(c);
  env->ReleaseStringUTFChars(val, c);
  return s;
}

std::string nativeLibDir(android_app* app, JNIEnv* env) {
  jobject activity = app->activity->clazz;
  jclass actCls = env->GetObjectClass(activity);
  jmethodID getAppInfo =
      env->GetMethodID(actCls, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
  jobject info = env->CallObjectMethod(activity, getAppInfo);
  jfieldID fid =
      env->GetFieldID(env->GetObjectClass(info), "nativeLibraryDir", "Ljava/lang/String;");
  auto dir = static_cast<jstring>(env->GetObjectField(info, fid));
  const char* c = env->GetStringUTFChars(dir, nullptr);
  std::string s(c);
  env->ReleaseStringUTFChars(dir, c);
  return s;
}

std::string materializeAsset(android_app* app, const char* name) {
  const std::string path = std::string(app->activity->internalDataPath) + "/" + name;
  AAsset* a = AAssetManager_open(app->activity->assetManager, name, AASSET_MODE_STREAMING);
  if (!a) { LOGE("asset %s missing", name); return ""; }
  const off_t want = AAsset_getLength(a);
  FILE* existing = fopen(path.c_str(), "rb");
  if (existing) {
    fseek(existing, 0, SEEK_END);
    const long have = ftell(existing);
    fclose(existing);
    if (have == want) { AAsset_close(a); return path; }
  }
  FILE* f = fopen(path.c_str(), "wb");
  char buf[65536];
  int n = 0;
  while ((n = AAsset_read(a, buf, sizeof(buf))) > 0) fwrite(buf, 1, static_cast<size_t>(n), f);
  fclose(f);
  AAsset_close(a);
  return path;
}

}  // namespace mlvc
