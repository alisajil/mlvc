#pragma once
// Receiver mode: no camera, no HUD - this app instance owns the whole window
// and spends it on the decoded picture. See decode_loop.cpp for the full
// SRT-receive -> rANS-decode -> NPU-decode -> render pipeline.

#include <android_native_app_glue.h>

#include <string>

namespace mlvc {

// Blocks until one SRT sender connects, then decodes and renders frames into
// app->window until the sender disconnects. 720p only (rejects any other
// resolution in the header) - matches this build's compiled decoder model.
// `bindAddr` should be the exact address the sender dials (empty = wildcard
// "any") - on a multi-address host, a wildcard bind lets the OS pick its own
// reply source address, and when that differs from what the sender targeted,
// the sender silently drops the handshake response (see srtAcceptOne).
void decodeLoop(android_app* app, int listenPort, const std::string& decPath,
                const std::string& pmfPath, const std::string& backendLib,
                int srtLatencyMs, const std::string& bindAddr = "");

}  // namespace mlvc
