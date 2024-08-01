#include "layernorm.h"

#include <arm_neon.h>
#include <cstdio>

void layernorm_A(Matmul *x, int rows, int cols,
                 const std::vector<__fp16> &gamma,
                 const std::vector<__fp16> &beta, float eps)
{
  assert(!(cols % 8));
#pragma omp parallel for
  for (int i = 0; i < rows; i++)
  {
    // Calculate mean.
    __fp16 *input_ptr = x->get_A_ptr() + 8 * i;
    float16x8_t mean = vdupq_n_f16(0);
    float32x4_t mean_f32x4_high = vdupq_n_f32(0);
    float32x4_t mean_f32x4_low = vdupq_n_f32(0);

    for (int j = 0; j < cols; j += 8)
    {
      int offset = rows * j;
      // mean = vaddq_f16(vld1q_f16(input_ptr + offset), mean);
      float16x8_t cur_8 = vld1q_f16(input_ptr + offset);
      mean_f32x4_high = vaddq_f32(vcvt_f32_f16(vget_high_f16(cur_8)),mean_f32x4_high);
      mean_f32x4_low = vaddq_f32(vcvt_f32_f16(vget_low_f16(cur_8)),mean_f32x4_low);
    }
    
    // float16x4_t mean_f16x4 = vadd_f16(vget_high_f16(mean), vget_low_f16(mean));
    // float32x4_t mean_f32x4 = vcvt_f32_f16(mean_f16x4);

    // mean_f32x4_high = vcvt_f32_f16(vget_high_f16(mean));
    // mean_f32x4_low = vcvt_f32_f16(vget_low_f16(mean));
    float32x4_t mean_f32x4 = vaddq_f32(mean_f32x4_high, mean_f32x4_low);

    float32_t mean_f32 = vaddvq_f32(mean_f32x4) / (float32_t)cols;
    // //判断mean的值是否超过65503
    // if(std::isnan(mean_f32) || std::isinf(mean_f32) ||std::fabs(mean_f32)>65503)
    // {
    //   printf("layernorm_A mean3 >65503 \n");
    // }
    mean = vdupq_n_f16((__fp16)mean_f32);
    mean_f32x4_high = vdupq_n_f32(mean_f32);
    mean_f32x4_low = vdupq_n_f32(mean_f32);

    // Calculate mean squared deviation
    // float16x8_t msd = vdupq_n_f16(0);
    float32x4_t msd_f32x4_high = vdupq_n_f32(0);
    float32x4_t msd_f32x4_low = vdupq_n_f32(0);
    for (int j = 0; j < cols; j += 8)
    {
      int offset = rows * j;
      float16x8_t val = vld1q_f16(input_ptr + offset);
      float32x4_t val_f32x4_high = vcvt_f32_f16(vget_high_f16(val));
      float32x4_t val_f32x4_low = vcvt_f32_f16(vget_low_f16(val));
      val_f32x4_high = vsubq_f32(val_f32x4_high, mean_f32x4_high);
      val_f32x4_low = vsubq_f32(val_f32x4_low, mean_f32x4_low);
      msd_f32x4_high = vaddq_f32(vmulq_f32(val_f32x4_high, val_f32x4_high), msd_f32x4_high);
      msd_f32x4_low = vaddq_f32(vmulq_f32(val_f32x4_low, val_f32x4_low), msd_f32x4_low);

      // val = vsubq_f16(val, mean);
      // val = vmulq_f16(val, val);
      // msd = vaddq_f16(val, msd);
    }

    // float16x4_t msd_f16x4 = vadd_f16(vget_high_f16(msd), vget_low_f16(msd));
    // float32x4_t msd_f32x4 = vcvt_f32_f16(msd_f16x4);
    float32x4_t msd_f32x4 = vaddq_f32(msd_f32x4_high, msd_f32x4_low);
    float32_t msd_f32 = vaddvq_f32(msd_f32x4);
    float32_t denom_single = msd_f32 / (float32_t)(cols - 1);
    denom_single = std::sqrt(denom_single + eps);
    float16x8_t denom = vdupq_n_f16((__fp16)denom_single);

    // Normalize based on mean + MSD, beta, gamma and epsilon.
    for (int j = 0; j < cols; j += 8)
    {
      int offset = rows * j;
      float16x8_t val = vld1q_f16(input_ptr + offset);
      float16x8_t gamma_val = vld1q_f16(gamma.data() + j);
      float16x8_t beta_val = vld1q_f16(beta.data() + j);
      val = vsubq_f16(val, mean);
      val = vdivq_f16(val, denom);
      val = vmulq_f16(val, gamma_val);
      val = vaddq_f16(val, beta_val);
      vst1q_f16(input_ptr + offset, val);
    }
  }
}
