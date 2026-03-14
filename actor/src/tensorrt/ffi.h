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

typedef struct TrtRuntime TrtRuntime;
typedef struct TrtEngine TrtEngine;
typedef struct TrtContext TrtContext;

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

#ifdef __cplusplus
}
#endif

#endif
