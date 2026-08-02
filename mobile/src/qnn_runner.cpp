#include "qnn_runner.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

namespace mlvc {
namespace {

// Tensor accessors that tolerate both v1 and v2 tensor structs.
const char* tName(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
}
uint32_t tRank(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
const uint32_t* tDims(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
Qnn_DataType_t tType(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
void tSetClientBuf(Qnn_Tensor_t& t, void* data, uint32_t size) {
  if (t.version == QNN_TENSOR_VERSION_2) {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf.data = data;
    t.v2.clientBuf.dataSize = size;
  } else {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = data;
    t.v1.clientBuf.dataSize = size;
  }
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::vector<char> buf(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  return f ? buf : std::vector<char>{};
}

}  // namespace

size_t qnnElementSize(uint32_t dataType) {
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      return 4;
    case QNN_DATATYPE_FLOAT_64:
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      return 8;
    default:
      return 1;  // int8/uint8/bool_8 and quantized 8-bit variants
  }
}

struct QnnRunner::Impl {
  void* backendLib = nullptr;
  void* systemLib = nullptr;
  QNN_INTERFACE_VER_TYPE fn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sysFn{};
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ContextHandle_t context = nullptr;
  QnnSystemContext_Handle_t sysCtx = nullptr;
  std::unordered_map<std::string, Qnn_GraphHandle_t> graphHandles;
  // Per-graph deep-copied tensor structs ready for execute().
  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>> inTensors;
  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>> outTensors;
  // Backing store for dims arrays + names referenced by the copied tensors.
  std::vector<std::vector<uint32_t>> dimStore;
  std::vector<std::string> nameStore;
  uint32_t powerConfigId = 0;
  bool powerConfigSet = false;
  QnnHtpDevice_PerfInfrastructure_t perfInfra{};
};

QnnRunner::~QnnRunner() {
  if (!impl_) return;
  if (impl_->powerConfigSet && impl_->perfInfra.destroyPowerConfigId) {
    impl_->perfInfra.destroyPowerConfigId(impl_->powerConfigId);
  }
  if (impl_->sysCtx && impl_->sysFn.systemContextFree) {
    impl_->sysFn.systemContextFree(impl_->sysCtx);
  }
  if (impl_->context) impl_->fn.contextFree(impl_->context, nullptr);
  if (impl_->device && impl_->fn.deviceFree) impl_->fn.deviceFree(impl_->device);
  if (impl_->backend) impl_->fn.backendFree(impl_->backend);
  if (impl_->log && impl_->fn.logFree) impl_->fn.logFree(impl_->log);
  // Deliberately no dlclose: libQnnHtp leaves DSP worker threads alive and
  // unloading the library under them segfaults at exit. Process teardown
  // reclaims everything.
  delete impl_;
}

bool QnnRunner::init(const std::string& backendPath, const std::string& systemPath,
                     const std::string& contextPath, bool burstMode) {
  impl_ = new Impl();

#define FAILF(...)                       \
  do {                                   \
    char msg[512];                       \
    snprintf(msg, sizeof(msg), __VA_ARGS__); \
    error_ = msg;                        \
    return false;                        \
  } while (0)

  // --- backend interface ---
  impl_->backendLib = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!impl_->backendLib) FAILF("dlopen %s: %s", backendPath.c_str(), dlerror());
  using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(impl_->backendLib, "QnnInterface_getProviders"));
  if (!getProviders) FAILF("QnnInterface_getProviders not found");
  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    FAILF("no QNN interface providers");
  }
  impl_->fn = providers[0]->QNN_INTERFACE_VER_NAME;

  // --- system interface (context binary introspection) ---
  impl_->systemLib = dlopen(systemPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!impl_->systemLib) FAILF("dlopen %s: %s", systemPath.c_str(), dlerror());
  using GetSysProvidersFn =
      Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
  auto getSysProviders = reinterpret_cast<GetSysProvidersFn>(
      dlsym(impl_->systemLib, "QnnSystemInterface_getProviders"));
  if (!getSysProviders) FAILF("QnnSystemInterface_getProviders not found");
  const QnnSystemInterface_t** sysProviders = nullptr;
  uint32_t numSys = 0;
  if (getSysProviders(&sysProviders, &numSys) != QNN_SUCCESS || numSys == 0) {
    FAILF("no QNN system providers");
  }
  impl_->sysFn = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

  // --- backend + device ---
  if (impl_->fn.logCreate) {
    impl_->fn.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &impl_->log);
  }
  if (impl_->fn.backendCreate(impl_->log, nullptr, &impl_->backend) != QNN_SUCCESS) {
    FAILF("backendCreate failed");
  }
  // HTP accepts a default device; failure here is fatal only if contextCreate
  // later also fails, so try with device first and retry with null.
  if (impl_->fn.deviceCreate &&
      impl_->fn.deviceCreate(impl_->log, nullptr, &impl_->device) != QNN_SUCCESS) {
    impl_->device = nullptr;
  }

  // --- sustained perf mode (was: pinned-max "burst", benchmark parity with
  // qnn-net-run - appropriate for a one-shot profiling run, wrong for a live
  // stream that has to run for minutes: pinning both voltage corners to MAX
  // and disabling DSP sleep between calls kept the chip at peak draw
  // continuously, even during the ~10ms/frame idle gap our real workload
  // has (measured total ~20ms of a 33ms budget). POWER_SAVER_MODE lets DCVS
  // actually scale down when idle and ramp up under real load, instead of
  // sitting pinned high regardless of demand - same correctness, same frame
  // budget headroom, far less sustained heat. ponytail: TURBO ceiling chosen
  // for headroom over our measured ~20ms/frame; lower it further if thermal
  // margin needs to grow, raise it if a heavier model needs the ceiling.
  if (burstMode && impl_->fn.deviceGetInfrastructure) {
    QnnDevice_Infrastructure_t devInfra = nullptr;
    if (impl_->fn.deviceGetInfrastructure(&devInfra) == QNN_SUCCESS && devInfra) {
      auto* htpInfra = static_cast<QnnHtpDevice_Infrastructure_t*>(devInfra);
      if (htpInfra->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        impl_->perfInfra = htpInfra->perfInfra;
        if (impl_->perfInfra.createPowerConfigId(0, 0, &impl_->powerConfigId) ==
            QNN_SUCCESS) {
          QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
          dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
          dcvs.dcvsV3Config.contextId = impl_->powerConfigId;
          dcvs.dcvsV3Config.setDcvsEnable = 1;
          dcvs.dcvsV3Config.dcvsEnable = 1;  // let clocks scale with real load
          dcvs.dcvsV3Config.powerMode =
              QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
          dcvs.dcvsV3Config.setSleepLatency = 1;
          dcvs.dcvsV3Config.sleepLatency = 40;
          dcvs.dcvsV3Config.setSleepDisable = 1;
          dcvs.dcvsV3Config.sleepDisable = 0;  // allow DSP sleep in the idle gap each frame
          dcvs.dcvsV3Config.setBusParams = 1;
          dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
          dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
          dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
          dcvs.dcvsV3Config.setCoreParams = 1;
          dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
          dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
          dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
          const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, nullptr};
          if (impl_->perfInfra.setPowerConfig(impl_->powerConfigId, cfgs) ==
              QNN_SUCCESS) {
            impl_->powerConfigSet = true;
          }
        }
      }
    }
    if (!impl_->powerConfigSet) {
      fprintf(stderr, "warn: sustained perf config not applied, running default clocks\n");
    }
  }

  // --- context binary ---
  std::vector<char> bin = readFile(contextPath);
  if (bin.empty()) FAILF("cannot read %s", contextPath.c_str());

  // Introspect I/O specs before creating the executable context.
  if (impl_->sysFn.systemContextCreate(&impl_->sysCtx) != QNN_SUCCESS) {
    FAILF("systemContextCreate failed");
  }
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (impl_->sysFn.systemContextGetBinaryInfo(
          impl_->sysCtx, bin.data(), bin.size(), &info, &infoSize) != QNN_SUCCESS ||
      !info) {
    FAILF("getBinaryInfo failed (wrong QNN version for this binary?)");
  }

  const QnnSystemContext_GraphInfo_t* graphsArr = nullptr;
  uint32_t numGraphs = 0;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphsArr = info->contextBinaryInfoV1.graphs;
    numGraphs = info->contextBinaryInfoV1.numGraphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphsArr = info->contextBinaryInfoV2.graphs;
    numGraphs = info->contextBinaryInfoV2.numGraphs;
  } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphsArr = info->contextBinaryInfoV3.graphs;
    numGraphs = info->contextBinaryInfoV3.numGraphs;
  } else {
    FAILF("unsupported binary info version %d", static_cast<int>(info->version));
  }

  for (uint32_t g = 0; g < numGraphs; ++g) {
    const char* gname = nullptr;
    const Qnn_Tensor_t* gIn = nullptr;
    const Qnn_Tensor_t* gOut = nullptr;
    uint32_t nIn = 0, nOut = 0;
    const auto& gi = graphsArr[g];
    if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      gname = gi.graphInfoV1.graphName;
      gIn = gi.graphInfoV1.graphInputs;   nIn = gi.graphInfoV1.numGraphInputs;
      gOut = gi.graphInfoV1.graphOutputs; nOut = gi.graphInfoV1.numGraphOutputs;
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
      gname = gi.graphInfoV2.graphName;
      gIn = gi.graphInfoV2.graphInputs;   nIn = gi.graphInfoV2.numGraphInputs;
      gOut = gi.graphInfoV2.graphOutputs; nOut = gi.graphInfoV2.numGraphOutputs;
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      gname = gi.graphInfoV3.graphName;
      gIn = gi.graphInfoV3.graphInputs;   nIn = gi.graphInfoV3.numGraphInputs;
      gOut = gi.graphInfoV3.graphOutputs; nOut = gi.graphInfoV3.numGraphOutputs;
    } else {
      FAILF("unsupported graph info version %d", static_cast<int>(gi.version));
    }

    GraphIo io;
    io.graphName = gname;
    auto copyTensors = [&](const Qnn_Tensor_t* src, uint32_t n,
                           std::vector<TensorDesc>& descs,
                           std::vector<Qnn_Tensor_t>& copies) {
      for (uint32_t i = 0; i < n; ++i) {
        TensorDesc d;
        d.name = tName(src[i]);
        d.dims.assign(tDims(src[i]), tDims(src[i]) + tRank(src[i]));
        d.dataType = tType(src[i]);
        d.byteSize = qnnElementSize(d.dataType);
        for (uint32_t v : d.dims) d.byteSize *= v;
        descs.push_back(d);

        Qnn_Tensor_t copy = src[i];  // shallow copy, then re-point dims + name
        impl_->dimStore.push_back(d.dims);
        impl_->nameStore.push_back(d.name);
        if (copy.version == QNN_TENSOR_VERSION_2) {
          copy.v2.dimensions = impl_->dimStore.back().data();
          copy.v2.name = impl_->nameStore.back().c_str();
        } else {
          copy.v1.dimensions = impl_->dimStore.back().data();
          copy.v1.name = impl_->nameStore.back().c_str();
        }
        copies.push_back(copy);
      }
    };
    copyTensors(gIn, nIn, io.inputs, impl_->inTensors[io.graphName]);
    copyTensors(gOut, nOut, io.outputs, impl_->outTensors[io.graphName]);
    graphIo_.push_back(std::move(io));
  }

  // NOTE: dimStore growth invalidates earlier data() pointers only if the
  // vector reallocates; reserve enough up front to keep pointers stable.
  // (Descriptors were already collected above; re-point everything now.)
  // Simpler + safe: rebuild pointers after all pushes.
  {
    size_t k = 0;
    for (auto& g : graphIo_) {
      for (size_t i = 0; i < g.inputs.size(); ++i, ++k) {
        auto& t = impl_->inTensors[g.graphName][i];
        if (t.version == QNN_TENSOR_VERSION_2) {
          t.v2.dimensions = impl_->dimStore[k].data();
          t.v2.name = impl_->nameStore[k].c_str();
        } else {
          t.v1.dimensions = impl_->dimStore[k].data();
          t.v1.name = impl_->nameStore[k].c_str();
        }
      }
      for (size_t i = 0; i < g.outputs.size(); ++i, ++k) {
        auto& t = impl_->outTensors[g.graphName][i];
        if (t.version == QNN_TENSOR_VERSION_2) {
          t.v2.dimensions = impl_->dimStore[k].data();
          t.v2.name = impl_->nameStore[k].c_str();
        } else {
          t.v1.dimensions = impl_->dimStore[k].data();
          t.v1.name = impl_->nameStore[k].c_str();
        }
      }
    }
  }

  if (impl_->fn.contextCreateFromBinary(
          impl_->backend, impl_->device, nullptr, bin.data(), bin.size(),
          &impl_->context, nullptr) != QNN_SUCCESS) {
    FAILF("contextCreateFromBinary failed");
  }

  for (const auto& g : graphIo_) {
    Qnn_GraphHandle_t gh = nullptr;
    if (impl_->fn.graphRetrieve(impl_->context, g.graphName.c_str(), &gh) !=
        QNN_SUCCESS) {
      FAILF("graphRetrieve(%s) failed", g.graphName.c_str());
    }
    impl_->graphHandles[g.graphName] = gh;
  }
#undef FAILF
  return true;
}

bool QnnRunner::execute(const std::string& graphName,
                        const std::vector<void*>& inputBufs,
                        const std::vector<void*>& outputBufs) {
  auto it = impl_->graphHandles.find(graphName);
  if (it == impl_->graphHandles.end()) {
    error_ = "unknown graph " + graphName;
    return false;
  }
  const GraphIo* io = nullptr;
  for (const auto& g : graphIo_) {
    if (g.graphName == graphName) { io = &g; break; }
  }
  auto& ins = impl_->inTensors[graphName];
  auto& outs = impl_->outTensors[graphName];
  if (inputBufs.size() != ins.size() || outputBufs.size() != outs.size()) {
    error_ = "buffer count mismatch";
    return false;
  }
  for (size_t i = 0; i < ins.size(); ++i) {
    tSetClientBuf(ins[i], inputBufs[i], static_cast<uint32_t>(io->inputs[i].byteSize));
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    tSetClientBuf(outs[i], outputBufs[i], static_cast<uint32_t>(io->outputs[i].byteSize));
  }
  Qnn_ErrorHandle_t err = impl_->fn.graphExecute(
      it->second, ins.data(), static_cast<uint32_t>(ins.size()), outs.data(),
      static_cast<uint32_t>(outs.size()), nullptr, nullptr);
  if (err != QNN_SUCCESS) {
    error_ = "graphExecute failed code " + std::to_string(err);
    return false;
  }
  return true;
}

}  // namespace mlvc
