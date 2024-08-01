#ifndef _LIB_MISC_H_
#define _LIB_MISC_H_

union u16_f16 {
  uint16_t u;
  __fp16 f;
};

inline __fp16 to_f16(uint16_t u) {
  u16_f16 t{.u = u};
  return t.f;
}

inline uint16_t to_u16(__fp16 f) {
  u16_f16 t{.f = f};
  return t.u;
}

union u32_f32 {
  uint32_t u;
  float f;
};

inline float to_f32(uint32_t u) {
  u32_f32 t{.u = u};
  return t.f;
}

inline uint32_t to_u32(float f) {
  u32_f32 t{.f = f};
  return t.u;
}

#endif  // _LIB_MISC_H_
