#!/bin/bash
# Live 720p demo launcher: phone front camera -> NPU encode -> SRT/cellular ->
# Mac CoreML decode -> ffplay. Wraps the exact manual sequence used for live
# testing, plus a thermal guard - refuses to start if the phone is already
# hot, and auto-kills everything mid-run if it climbs into CRITICAL, instead
# of relying on someone checking dumpsys by hand.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ADB_SERIAL="${ADB_SERIAL:-$(adb devices | awk 'NR==2{print $1}')}"
SRT_PORT="${SRT_PORT:-8720}"
POLL_SECS=10

# THERMAL_STATUS_* per Android PowerManager: NONE=0 LIGHT=1 MODERATE=2
# SEVERE=3 CRITICAL=4 EMERGENCY=5 SHUTDOWN=6.
REFUSE_AT=3   # don't even start if already this hot
ABORT_AT=4    # kill an in-progress run if it climbs to this

adb_() { adb -s "$ADB_SERIAL" "$@"; }

thermal_status() {
  adb_ shell "dumpsys thermalservice 2>/dev/null" \
    | grep -m1 "^Thermal Status:" | grep -oE '[0-9]+'
}

cleanup() {
  echo "cleaning up..."
  [ -n "${MONITOR_PID:-}" ] && kill "$MONITOR_PID" 2>/dev/null
  # $DEC_PID is just the wrapper subshell for "mlvc_decode | ffplay" - killing
  # it doesn't cascade to both ends of the pipe, so pattern-match both
  # processes directly, scoped to this run's port to avoid hitting an
  # unrelated concurrent demo.
  pkill -f "mlvc_decode --listen $SRT_PORT " 2>/dev/null
  pkill -f "ffplay -f rawvideo -pixel_format yuv420p -video_size 1280x720" 2>/dev/null
  [ -n "${DEC_PID:-}" ] && kill "$DEC_PID" 2>/dev/null
  adb_ shell am force-stop com.mlvc.cam >/dev/null 2>&1
  adb_ shell "pkill -f mlvc_encode" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

if [ -z "$ADB_SERIAL" ]; then
  echo "no adb device found"; exit 1
fi

status=$(thermal_status)
if [ -z "$status" ]; then
  echo "WARNING: could not read thermal status, proceeding without pre-flight check"
elif [ "$status" -ge "$REFUSE_AT" ]; then
  echo "REFUSED: phone thermal status=$status (>= $REFUSE_AT SEVERE) - let it cool first"
  exit 1
else
  echo "thermal pre-flight OK: status=$status"
fi

MAC_ADDR=$(ifconfig en0 | awk '/inet6.*secured/ && !/fe80/ {print $2; exit}')
if [ -z "$MAC_ADDR" ]; then
  echo "could not resolve Mac secured IPv6 address on en0"; exit 1
fi
echo "Mac addr: $MAC_ADDR  SRT port: $SRT_PORT"

# Model selection: default is the psnr checkpoint; MODEL=perceptual switches
# both ends to the perceptual variant (phone context binaries + pmf must
# already be pushed, Mac CoreML bundle already exported).
MODEL="${MODEL:-psnr}"
if [ "$MODEL" = "perceptual" ]; then
  MAC_MODEL="/Users/sajil/genzee/mlvc/video/output/models/mlvc_s-mlvc-s-perceptual-v1/coreml-apple/1280x720/MLVCDecoder.mlpackage"
  PHONE_ENC1="perc_720p_enc1_v75.bin"; PHONE_ENC2="perc_720p_enc2_v75.bin"
  PHONE_PMF="pmf_tables_perceptual.bin"
  MAC_PMF="/Users/sajil/genzee/mlvc/demo_asset/pmf_tables_perceptual.bin"
else
  MAC_MODEL="/Users/sajil/genzee/mlvc/video/output/models/mlvc_s-mlvc-s-psnr-v1/coreml-apple/1280x720/MLVCDecoder.mlmodelc"
  PHONE_ENC1="s24_720p_enc1_v75.bin"; PHONE_ENC2="s24_720p_enc2_v75.bin"
  PHONE_PMF="pmf_tables.bin"
  MAC_PMF="$HERE/pmf_tables.bin"
fi
echo "model: $MODEL"

DEC_LOG="$HERE/live_dec.log"
DEC_QLOG_EXPORT=""
[ "${QLOG:-0}" = "1" ] && DEC_QLOG_EXPORT="MLVC_QLOG=1"
bash -c "$DEC_QLOG_EXPORT '$HERE/build_mac/mlvc_decode' --listen $SRT_PORT --public --srt --bind $MAC_ADDR \
  --model '$MAC_MODEL' \
  --pmf '$MAC_PMF' --out /dev/stdout 2>'$DEC_LOG' \
  | ffplay -f rawvideo -pixel_format yuv420p -video_size 1280x720 -framerate 30 -" &
DEC_PID=$!
sleep 1
echo "decoder: $(tail -2 "$DEC_LOG" 2>/dev/null)"

# ADAPTIVE=1 turns on the AIMD rate controller (--adaptive --q-min --q-max).
# The encoder-side plumbing runs over any transport, but the congestion
# signal it listens for (receiver wait-time) is shaped around TCP - SRT's
# TLPKTDROP turns congestion into an instant frame-skip rather than growing
# wait time, so it may never fire here in practice. See PLAN.md.
ADAPTIVE_ARGS=""
if [ "${ADAPTIVE:-0}" = "1" ]; then
  ADAPTIVE_ARGS="--adaptive --q-min ${Q_MIN:-21} --q-max ${Q_MAX:-63}"
  echo "adaptive rate control: on (q-min=${Q_MIN:-21} q-max=${Q_MAX:-63})"
fi

# getenv() only checks the pointer is non-null, so an *exported-but-empty*
# MLVC_QLOG would still count as "set" and force logging on by default -
# only export it at all when actually requested.
QLOG_EXPORT=""
[ "${QLOG:-0}" = "1" ] && QLOG_EXPORT="MLVC_QLOG=1"

adb_ shell "cd /data/local/tmp/mlvc && export LD_LIBRARY_PATH=/data/local/tmp/mlvc ADSP_LIBRARY_PATH=/data/local/tmp/mlvc $QLOG_EXPORT && nohup ./mlvc_encode --yuv-listen 8901 --width 1280 --height 720 --frames 0 --q 63 $ADAPTIVE_ARGS --enc1 $PHONE_ENC1 --enc2 $PHONE_ENC2 --pmf $PHONE_PMF --srt --stream '[$MAC_ADDR]:$SRT_PORT' > /data/local/tmp/mlvc/enc_live.log 2>&1 & disown" >/dev/null 2>&1 &
sleep 2
adb_ shell am start -n com.mlvc.cam/android.app.NativeActivity -e mode bridge -e camera front -e bridge_port 8901 >/dev/null

echo "live - thermal-monitoring every ${POLL_SECS}s (abort at status>=$ABORT_AT), Ctrl-C to stop"
(
  while true; do
    sleep "$POLL_SECS"
    s=$(thermal_status)
    [ -z "$s" ] && continue
    echo "[thermal] status=$s"
    if [ "$s" -ge "$ABORT_AT" ]; then
      echo "ABORT: thermal status=$s (>= $ABORT_AT CRITICAL) - stopping stream to let phone cool"
      kill -TERM $$ 2>/dev/null
      break
    fi
  done
) &
MONITOR_PID=$!

wait "$DEC_PID"
