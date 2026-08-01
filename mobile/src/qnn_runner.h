#pragma once
// Minimal QNN HTP runner: loads a context binary, exposes named graphs,
// executes with caller-provided host buffers.
// ponytail: single context, single device, synchronous execute. Multi-context
// composition or async pipelining only if the frame loop needs overlap.

#include <cstdint>
#include <string>
#include <vector>

namespace mlvc {

struct TensorDesc {
  std::string name;
  std::vector<uint32_t> dims;
  uint32_t dataType = 0;  // Qnn_DataType_t value
  size_t byteSize = 0;    // product(dims) * element size
};

struct GraphIo {
  std::string graphName;
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
};

class QnnRunner {
 public:
  QnnRunner() = default;
  ~QnnRunner();
  QnnRunner(const QnnRunner&) = delete;
  QnnRunner& operator=(const QnnRunner&) = delete;

  // Loads backend/system libs, parses the context binary and registers all
  // graphs. Returns false with a message in error() on any failure.
  bool init(const std::string& backendPath, const std::string& systemPath,
            const std::string& contextPath, bool burstMode);

  const std::vector<GraphIo>& graphs() const { return graphIo_; }
  const std::string& error() const { return error_; }

  // Executes graph by name. Buffer counts and byte sizes must match the
  // GraphIo descriptors exactly.
  bool execute(const std::string& graphName,
               const std::vector<void*>& inputBufs,
               const std::vector<void*>& outputBufs);

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  std::vector<GraphIo> graphIo_;
  std::string error_;
};

size_t qnnElementSize(uint32_t dataType);

}  // namespace mlvc
