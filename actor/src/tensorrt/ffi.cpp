#include "ffi.h"

#include <NvInferRuntime.h>

#include <cstring>
#include <exception>
#include <fstream>
#include <string>
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

struct TrtRuntime { nvinfer1::IRuntime* runtime; }
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
    if (rt) {
        delete rt->runtime;
        delete rt;
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
        auto* engine = rt->runtime->deserializeCudaEngine(data.data(), data.size());
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

}
