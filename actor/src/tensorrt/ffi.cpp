#include "ffi.h"

#include <NvInferRuntime.h>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <optional>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

static TrtLogger& get_logger() {
    static TrtLogger logger;
    return logger;
}

static thread_local std::string g_last_error;

static void set_error(const char* msg) { g_last_error = msg; }
static void set_error(const std::string& msg) { g_last_error = msg; }

struct TrtRuntime { nvinfer1::IRuntime* runtime; };
struct TrtEngine { nvinfer1::ICudaEngine* engine; };
struct TrtContext { nvinfer1::IExecutionContext* context; };

extern "C" {

const char* trt_get_last_error(void) {
    return g_last_error.c_str();
}

TrtStatus trt_runtime_create(TrtRuntime** out) {
    try {
        auto* runtime = nvinfer1::createInferRuntime(get_logger());
        if (!runtime) {
            set_error("Failed to create TensorRT runtime");
            return TRT_ERROR;
        }
        *out = new TrtRuntime{runtime};
        return TRT_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRT_ERROR;
    }
}

void trt_runtime_destroy(TrtRuntime* runtime) {
    if (runtime) {
        delete runtime->runtime;
        delete runtime;
    }
}

TrtStatus trt_engine_load(TrtRuntime* runtime, const char* path, TrtEngine** out) {
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            set_error(std::string("Failed to open engine file: ") + path);
            return TRT_ERROR;
        }
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        if (!file.read(data.data(), size)) {
            set_error(std::string("Failed to read engine file: ") + path);
            return TRT_ERROR;
        }
        auto* engine = runtime->runtime->deserializeCudaEngine(data.data(), data.size());
        if (!engine) {
            set_error("Failed to deserialize CUDA engine");
            return TRT_ERROR;
        }
        *out = new TrtEngine{engine};
        return TRT_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRT_ERROR;
    }
}

int32_t trt_engine_get_num_io_tensors(TrtEngine* engine) {
    return engine->engine->getNbIOTensors();
}

const char* trt_engine_get_io_tensor_name(TrtEngine* engine, int32_t index) {
    if (index < 0 || index >= engine->engine->getNbIOTensors()) {
        return nullptr;
    }
    return engine->engine->getIOTensorName(index);
}

int32_t trt_engine_get_tensor_io_mode(TrtEngine* engine, const char* name) {
    auto mode = engine->engine->getTensorIOMode(name);
    return (mode == nvinfer1::TensorIOMode::kINPUT) ? 0 : 1;
}

int32_t trt_engine_get_tensor_dtype(TrtEngine* engine, const char* name) {
    auto dtype = engine->engine->getTensorDataType(name);
    return static_cast<int32_t>(dtype);
}

int32_t trt_engine_get_tensor_shape(TrtEngine* engine, const char* name,
                                     int64_t* dims, int32_t capacity) {
    auto shape = engine->engine->getTensorShape(name);
    int32_t ndims = shape.nbDims;
    int32_t to_copy = (ndims < capacity) ? ndims : capacity;
    for (int32_t i = 0; i < to_copy; i++) {
        dims[i] = shape.d[i];
    }
    return ndims;
}

void trt_engine_destroy(TrtEngine* engine) {
    if (engine) {
        delete engine->engine;
        delete engine;
    }
}

TrtStatus trt_context_create(TrtEngine* engine, TrtContext** out) {
    try {
        auto* context = engine->engine->createExecutionContext();
        if (!context) {
            set_error("Failed to create execution context");
            return TRT_ERROR;
        }
        *out = new TrtContext{context};
        return TRT_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRT_ERROR;
    }
}

TrtStatus trt_context_set_input_shape(TrtContext* context, const char* name,
                                       const int64_t* dims, int32_t ndims) {
    nvinfer1::Dims d;
    d.nbDims = ndims;
    for (int32_t i = 0; i < ndims; i++) {
        d.d[i] = dims[i];
    }
    if (!context->context->setInputShape(name, d)) {
        set_error(std::string("Failed to set input shape for tensor: ") + name);
        return TRT_ERROR;
    }
    return TRT_OK;
}

TrtStatus trt_context_set_tensor_address(TrtContext* context, const char* name, void* ptr) {
    if (!context->context->setTensorAddress(name, ptr)) {
        set_error(std::string("Failed to set tensor address for: ") + name);
        return TRT_ERROR;
    }
    return TRT_OK;
}

TrtStatus trt_context_enqueue(TrtContext* context, void* stream) {
    if (!context->context->enqueueV3(static_cast<cudaStream_t>(stream))) {
        set_error("Failed to enqueue inference");
        return TRT_ERROR;
    }
    return TRT_OK;
}

void trt_context_destroy(TrtContext* context) {
    if (context) {
        delete context->context;
        delete context;
    }
}

struct TrtLlmExecutor {
    tle::Executor executor;
    TrtLlmExecutor(std::filesystem::path const& engine_dir,tle::ModelType model_type,tle::ExecutorConfig const& config) : executor(engine_dir, model_type, config) {}
};

TrtLlmStatus trtllm_executor_create(const char* engine_dir,TrtLlmModelType model_type,float kv_cache_fraction,int32_t max_beam_width,TrtLlmExecutor** out_handle) {
    try {
        initTrtLlmPlugins();
        tle::ModelType mt;
        switch (model_type) {
            case TRTLLM_DECODER_ONLY: mt = tle::ModelType::kDECODER_ONLY; break;
            case TRTLLM_ENCODER_ONLY: mt = tle::ModelType::kENCODER_ONLY; break;
            case TRTLLM_ENCODER_DECODER: mt = tle::ModelType::kENCODER_DECODER; break;
            default:
                set_error("Invalid model type");
                return TRTLLM_ERROR;
        }
        tle::KvCacheConfig kv_config;
        if (kv_cache_fraction > 0.0f) {
            kv_config.setFreeGpuMemoryFraction(kv_cache_fraction);
        }
        tle::ExecutorConfig exec_config(max_beam_width);
        exec_config.setKvCacheConfig(kv_config);
        *out_handle = new TrtLlmExecutor(engine_dir,mt,exec_config);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown error");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_enqueue(TrtLlmExecutor* executor,const int32_t* tokens,size_t n_tokens,int32_t max_new,float temperature,int32_t top_k,float top_p,float rep_penalty,int32_t end_id,int32_t pad_id,int streaming,uint64_t* out_req_id) {
    try {
        tle::VecTokens input_tokens(tokens, tokens + n_tokens);

        tle::SamplingConfig sampling;
        if (temperature > 0.0f) {
            sampling.setTemperature(temperature);
        }
        if (top_k > 0) {
            sampling.setTopK(top_k);
        }
        if (top_p > 0.0f) {
            sampling.setTopP(top_p);
        }
        if (rep_penalty > 0.0f) {
            sampling.setRepetitionPenalty(rep_penalty);
        }

        tle::OutputConfig output_config;
        output_config.excludeInputFromOutput = true;

        std::optional<int32_t> opt_end_id;
        if (end_id >= 0) opt_end_id = end_id;

        std::optional<int32_t> opt_pad_id;
        if (pad_id >= 0) opt_pad_id = pad_id;

        tle::Request request(
            std::move(input_tokens),
            max_new,
            streaming != 0,
            sampling,
            output_config,
            opt_end_id,
            opt_pad_id);

        *out_req_id = executor->executor.enqueueRequest(request);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_enqueue");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_await(TrtLlmExecutor* executor,uint64_t req_id,int32_t* out_tokens,size_t capacity,size_t* out_n_tokens,int* out_is_final,int64_t timeout_ms) {
    try {
        std::optional<std::chrono::milliseconds> timeout;
        if (timeout_ms > 0) {
            timeout = std::chrono::milliseconds(timeout_ms);
        }

        auto responses = executor->executor.awaitResponses(req_id, timeout);

        if (responses.empty()) {
            *out_n_tokens = 0;
            *out_is_final = 0;
            return TRTLLM_OK;
        }

        // Take the last response (most recent).
        auto& response = responses.back();

        if (response.hasError()) {
            set_error(response.getErrorMsg());
            return TRTLLM_ERROR;
        }

        auto result = response.getResult();
        *out_is_final = result.isFinal ? 1 : 0;

        // Copy beam-0 output tokens.
        const auto& beam0 = result.outputTokenIds.at(0);
        size_t n = beam0.size();
        if (n > capacity) n = capacity;

        std::memcpy(out_tokens, beam0.data(), n * sizeof(int32_t));
        *out_n_tokens = n;

        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_await");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_cancel(TrtLlmExecutor* executor, uint64_t req_id) {
    try {
        executor->executor.cancelRequest(req_id);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_cancel");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_shutdown(TrtLlmExecutor* executor) {
    try {
        executor->executor.shutdown();
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_shutdown");
        return TRTLLM_ERROR;
    }
}

void trtllm_executor_destroy(TrtLlmExecutor* executor) {
    delete executor;
}

}
