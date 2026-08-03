#!/bin/bash
# Packages libmlvccam.so + QNN runtime libs + model assets into a signed APK.
# No gradle: aapt2 link + zip + zipalign + apksigner (hasCode=false, no dex).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDK=/opt/homebrew/share/android-commandlinetools
BT=$SDK/build-tools/36.0.0
PLATFORM=$SDK/platforms/android-34/android.jar
QAIRT=$HOME/mlvc/qairt/qairt/2.45.0.260326
MODELS=$HERE/../video/output/models/mlvc_s-mlvc-s-psnr-v1
STAGE=$HERE/build_apk_stage
OUT=$HERE/mlvccam.apk
KEYSTORE=$HERE/debug.keystore

rm -rf "$STAGE"
mkdir -p "$STAGE/lib/arm64-v8a" "$STAGE/assets"

cp "$HERE/build/libmlvccam.so" "$STAGE/lib/arm64-v8a/"
for so in libQnnHtp.so libQnnHtpV75Stub.so libQnnSystem.so libQnnCpu.so; do
  cp "$QAIRT/lib/aarch64-android/$so" "$STAGE/lib/arm64-v8a/"
done
cp "$QAIRT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "$STAGE/lib/arm64-v8a/"

cp "$HERE/../video/output/context_binaries/s24_720p_enc1_v75.bin" "$STAGE/assets/"
cp "$HERE/../video/output/context_binaries/s24_720p_enc2_v75.bin" "$STAGE/assets/"
cp "$HERE/../video/output/context_binaries/s24_720p_dec_v75.bin" "$STAGE/assets/"
cp "$HERE/pmf_tables.bin" "$STAGE/assets/"

"$BT/aapt2" compile --dir "$HERE/app/res" -o "$STAGE/compiled_res.zip"
"$BT/aapt2" link -o "$STAGE/base.apk" --manifest "$HERE/app/AndroidManifest.xml" -I "$PLATFORM" \
  -R "$STAGE/compiled_res.zip"

(cd "$STAGE" && zip -q -r base.apk lib assets)
"$BT/zipalign" -f 4 "$STAGE/base.apk" "$STAGE/aligned.apk"

if [ ! -f "$KEYSTORE" ]; then
  keytool -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android \
    -alias debug -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=MLVC Debug" > /dev/null 2>&1
fi
"$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
  --out "$OUT" "$STAGE/aligned.apk"

echo "built $OUT ($(du -h "$OUT" | cut -f1))"
