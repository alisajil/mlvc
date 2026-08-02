# Submit MLVC-S ONNX bundles to Qualcomm AI Hub for compile+profile on real devices.
import qai_hub as hub

BUNDLE = "output/models/mlvc_s-mlvc-s-psnr-v1/onnx-qualcomm"

# (job-name, onnx path)
MODELS_640 = [
    ("mlvc-s-enc1-640x368", f"{BUNDLE}/640x368/MLVCEncoderPart1.onnx"),
    ("mlvc-s-enc2-640x368", f"{BUNDLE}/640x368/MLVCEncoderPart2.onnx"),
    ("mlvc-s-dec-640x368", f"{BUNDLE}/640x368/MLVCDecoder.onnx"),
]
MODELS_320 = [
    ("mlvc-s-enc1-320x192", f"{BUNDLE}/320x192/MLVCEncoderPart1.onnx"),
    ("mlvc-s-enc2-320x192", f"{BUNDLE}/320x192/MLVCEncoderPart2.onnx"),
    ("mlvc-s-dec-320x192", f"{BUNDLE}/320x192/MLVCDecoder.onnx"),
]
# 1080p (padded 1920x1088; model crops back to 1080 on decode). de-risk run for
# full-HD streaming: same 8 Gen 3/v75 chip as the working 720p demo phone.
MODELS_1080 = [
    ("mlvc-s-enc1-1920x1088", f"{BUNDLE}/1920x1088/MLVCEncoderPart1.onnx"),
    ("mlvc-s-enc2-1920x1088", f"{BUNDLE}/1920x1088/MLVCEncoderPart2.onnx"),
    ("mlvc-s-dec-1920x1088", f"{BUNDLE}/1920x1088/MLVCDecoder.onnx"),
]

PLAN = [
    # 640x368 on mid-range + ceiling
    (MODELS_640, hub.Device("Snapdragon 7 Gen 4 QRD")),
    (MODELS_640, hub.Device("Samsung Galaxy S23")),
    # 320x192 on mid-range + old mid-range
    (MODELS_320, hub.Device("Snapdragon 7 Gen 4 QRD")),
    (MODELS_320, hub.Device("Samsung Galaxy A73 5G")),
    # 1080p de-risk: exact device used for the live 720p demo
    (MODELS_1080, hub.Device("Samsung Galaxy S24 Ultra")),
]

OPTIONS = "--target_runtime qnn_context_binary"

submitted = []
for models, device in PLAN:
    for name, path in models:
        try:
            jobs = hub.submit_compile_and_profile_jobs(
                model=path,
                device=device,
                name=name,
                compile_options=OPTIONS,
            )
            compile_job, profile_job = jobs
            submitted.append((name, device.name, compile_job.job_id,
                              profile_job.job_id if profile_job else "n/a"))
            print(f"OK  {name:24s} {device.name:28s} compile={compile_job.job_id} profile={profile_job.job_id if profile_job else 'n/a'}")
        except Exception as e:
            print(f"ERR {name:24s} {device.name:28s} {type(e).__name__}: {e}")

print(f"\n{len(submitted)} job pairs submitted. Dashboard: https://app.aihub.qualcomm.com/jobs/")
