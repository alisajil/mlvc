# One-off: compile the existing 1280x720 PSNR bundle for the closest AI Hub
# chipset match to the Xiaomi SM7435 (Snapdragon 7s Gen 2) test phone - AI
# Hub has no exact SM7435 target, "Snapdragon 7 Gen 4 QRD" is the nearest
# available and was already used successfully in this project for the
# 640x368/320x192 mid-range batch. Whether a v-whatever Hexagon binary
# compiled against 7 Gen 4 actually loads on the real SM7435 DSP is the
# open question this job (plus an on-device push/test) answers.
import qai_hub as hub

BUNDLE = "output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm/1280x720"

MODELS = [
    ("mlvc-s-enc1-720p-7gen4", f"{BUNDLE}/MLVCEncoderPart1.onnx"),
    ("mlvc-s-enc2-720p-7gen4", f"{BUNDLE}/MLVCEncoderPart2.onnx"),
    ("mlvc-s-dec-720p-7gen4", f"{BUNDLE}/MLVCDecoder.onnx"),
]

DEVICE = hub.Device("Snapdragon 7 Gen 4 QRD", os="15")
OPTIONS = "--target_runtime qnn_context_binary"

submitted = []
for name, path in MODELS:
    try:
        jobs = hub.submit_compile_and_profile_jobs(
            model=path, device=DEVICE, name=name, compile_options=OPTIONS,
        )
        compile_job, profile_job = jobs
        submitted.append((name, compile_job.job_id, profile_job.job_id if profile_job else "n/a"))
        print(f"OK  {name:28s} compile={compile_job.job_id} profile={profile_job.job_id if profile_job else 'n/a'}")
    except Exception as e:
        print(f"ERR {name:28s} {type(e).__name__}: {e}")

print(f"\n{len(submitted)} job pairs submitted. Dashboard: https://app.aihub.qualcomm.com/jobs/")
