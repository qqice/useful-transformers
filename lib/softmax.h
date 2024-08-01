#ifndef _LIB_SOFTMAX_H_
#define _LIB_SOFTMAX_H_

#include <vector>

#include "matmul.h"

using float32_t = float;
void softmax_C_to_A(Matmul* src, Matmul* dst, int rows, int cols);

// Copy from C operand of src to dst
void copy_C_to_fp16(Matmul* src, __fp16* dst, int rows, int cols);
__fp16 compute_max(__fp16* src, int N);
void log_softmax(__fp16* src, int N, __fp16 max);
void log_softmax32(__fp16* src, float32_t* target, int N, __fp16 max);
void copy_C_to_fp16and32(Matmul *src, __fp16 *dst1, float32_t *dst2, int rows, int cols);
void copy_C_to_fp32(Matmul* src, float32_t* dst, int rows, int cols);
// void log_softmax32(float32_t* src, int N, float32_t max);
#endif  // _LIB_SOFTMAX_H_
