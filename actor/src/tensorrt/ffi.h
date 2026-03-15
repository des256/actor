#ifndef _TENSORRT_FFI_H_
#define _TENSORRT_FFI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRT_OK = 0,
    TRT_ERROR = 1,
} TrtStatus;

typedef enum {
    TRTLLM_DECODER_ONLY = 0,
    TRTLLM_ENCODER_ONLY = 1,
    TRTLLM_ENCODER_DECODER = 2,
} TrtLlmModelType;

typedef struct TrtRuntime TrtRuntime;
typedef struct TrtEngine TrtEngine;
typedef struct TrtContext TrtContext;
typedef struct TrtLlmExecutor TrtLlmExecutor;

const char* trt_get_last_error(void);

TrtStatus trt_runtime_create(TrtRuntime** out);
void trt_runtime_destroy(TrtRuntime* runtime);
TrtStatus trt_engine_load(TrtRuntime* runtime,const char* path,TrtEngine** out);
int32_t trt_engine_get_num_io_tensors(TrtEngine* engine);
const char* trt_engine_get_io_tensor_name(TrtEngine* engine,int32_t index);
int32_t trt_engine_get_tensor_io_mode(TrtEngine* engine,const char* name);
int32_t trt_engine_get_tensor_dtype(TrtEngine* engine,const char* name);
int32_t trt_engine_get_tensor_shape(TrtEngine* engine,const char* name,int64_t* dims,int32_t capacity);
void trt_engine_destroy(TrtEngine* engine);
TrtStatus trt_context_create(TrtEngine* engine,TrtContext** out);
TrtStatus trt_context_set_input_shape(TrtContext* context,const char* name,const int64_t* dims,int32_t ndims);
TrtStatus trt_context_set_tensor_address(TrtContext* context,const char* name,void* ptr);
TrtStatus trt_context_enqueue(TrtContext* context,void* stream);
void trt_context_destroy(TrtContext* context);

TrtLlmStatus trtllm_executor_create(const char* engine_dir,TrtLlmModelType model_type,float kv_cache_fraction,int32_t max_beam_width,TrtLlmExecutor** out_handle);
TrtLlmStatus trtllm_executor_enqueue(TrtLlmExecutor* executor,const int32_t* tokens,size_t n_tokens,int32_t max_new,float temperature,int32_t top_k,float top_p,float rep_penalty,int32_t end_id,int32_t pad_id,int streaming,uint64_t* out_req_id);
TrtLlmStatus trtllm_executor_await(TrtLlmExecutor* executor,uint64_t req_id,int32_t* out_tokens,size_t capacity,size_t* out_n_tokens,int* out_is_final,int64_t timeout_ms);
TrtLlmStatus trtllm_executor_cancel(TrtLlmExecutor* executor, uint64_t req_id);
TrtLlmStatus trtllm_executor_shutdown(TrtLlmExecutor* executor);
void trtllm_executor_destroy(TrtLlmExecutor* executor);

#ifdef __cplusplus
}
#endif

#endif
