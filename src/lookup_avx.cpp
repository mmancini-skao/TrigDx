#include <algorithm>
#include <cmath>
#include <vector>

#include <immintrin.h>

#include "trigdx/lookup_avx.hpp"

#if defined(HAVE_AVX) && !defined(__AVX__)
static_assert(HAVE_AVX == 0, "__AVX__ should be defined when HAVE_AVX is defined, check compile flags (-mavx)");
#endif

#if defined(HAVE_AVX2) && !defined(__AVX2__)
static_assert(HAVE_AVX2 == 0, "__AVX__2 should be defined when HAVE_AVX2 is defined, check compile flags (-mavx2)");
#endif

template <std::size_t NR_SAMPLES> struct LookupAVXBackend<NR_SAMPLES>::Impl {
  std::vector<float> lookup;
  static constexpr std::size_t MASK = NR_SAMPLES - 1;
  static constexpr float SCALE = NR_SAMPLES / (2.0f * float(M_PI));

  void init() {
    lookup.resize(NR_SAMPLES);
    for (std::size_t i = 0; i < NR_SAMPLES; ++i) {
      lookup[i] = sinf(i * (2.0f * float(M_PI) / NR_SAMPLES));
    }
  }

  void compute_sincosf(std::size_t n, const float *x, float *s,
                       float *c) const {
#if defined(__AVX__)
    constexpr std::size_t VL = 8; // AVX processes 8 floats
    const __m256 scale = _mm256_set1_ps(SCALE);
    const __m256i mask = _mm256_set1_epi32(MASK);
    const __m256i quarter_pi = _mm256_set1_epi32(NR_SAMPLES / 4);

    std::size_t i = 0;
    for (; i + VL <= n; i += VL) {
      __m256 vx = _mm256_loadu_ps(&x[i]);
      __m256 scaled = _mm256_mul_ps(vx, scale);
      __m256i idx = _mm256_cvtps_epi32(scaled);
      __m256i idx_cos = _mm256_add_epi32(idx, quarter_pi);

      idx = _mm256_and_si256(idx, mask);
      idx_cos = _mm256_and_si256(idx_cos, mask);

#if defined(__AVX2__)
      __m256 sinv = _mm256_i32gather_ps(lookup.data(), idx, 4);
      __m256 cosv = _mm256_i32gather_ps(lookup.data(), idx_cos, 4);
#else
      // fallback gather for AVX1
      float sin_tmp[VL], cos_tmp[VL];
      int idx_a[VL], idxc_a[VL];
      _mm256_store_si256((__m256i *)idx_a, idx);
      _mm256_store_si256((__m256i *)idxc_a, idx_cos);
      for (std::size_t k = 0; k < VL; ++k) {
        sin_tmp[k] = lookup[idx_a[k]];
        cos_tmp[k] = lookup[idxc_a[k]];
      }
      __m256 sinv = _mm256_load_ps(sin_tmp);
      __m256 cosv = _mm256_load_ps(cos_tmp);
#endif
      _mm256_storeu_ps(&s[i], sinv);
      _mm256_storeu_ps(&c[i], cosv);
    }

    // scalar remainder
    for (; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      std::size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      s[i] = lookup[idx];
      c[i] = lookup[idx_cos];
    }
#else
    // No AVX: scalar path
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      std::size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      s[i] = lookup[idx];
      c[i] = lookup[idx_cos];
    }
#endif
  }

  void compute_sinf(std::size_t n, const float *x, float *s) const {
#if defined(__AVX__)
    constexpr std::size_t VL = 8; // AVX processes 8 floats
    const __m256 scale = _mm256_set1_ps(SCALE);
    const __m256i mask = _mm256_set1_epi32(MASK);
    const __m256i quarter_pi = _mm256_set1_epi32(NR_SAMPLES / 4);

    std::size_t i = 0;
    for (; i + VL <= n; i += VL) {
      __m256 vx = _mm256_loadu_ps(&x[i]);
      __m256 scaled = _mm256_mul_ps(vx, scale);
      __m256i idx = _mm256_cvtps_epi32(scaled);

      idx = _mm256_and_si256(idx, mask);

#if defined(__AVX2__)
      __m256 sinv = _mm256_i32gather_ps(lookup.data(), idx, 4);
#else
      // fallback gather for AVX1
      float sin_tmp[VL];
      int idx_a[VL], idxc_a[VL];
      _mm256_store_si256((__m256i *)idx_a, idx);
      for (std::size_t k = 0; k < VL; ++k) {
        sin_tmp[k] = lookup[idx_a[k]];
      }
      __m256 sinv = _mm256_load_ps(sin_tmp);
#endif
      _mm256_storeu_ps(&s[i], sinv);
    }

    // scalar remainder
    for (; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      s[i] = lookup[idx];
    }
#else
    // No AVX: scalar path
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      s[i] = lookup[idx];
    }
#endif
  }

  void compute_cosf(std::size_t n, const float *x, float *c) const {
#if defined(__AVX__)
    constexpr std::size_t VL = 8; // AVX processes 8 floats
    const __m256 scale = _mm256_set1_ps(SCALE);
    const __m256i mask = _mm256_set1_epi32(MASK);
    const __m256i quarter_pi = _mm256_set1_epi32(NR_SAMPLES / 4);

    std::size_t i = 0;
    for (; i + VL <= n; i += VL) {
      __m256 vx = _mm256_loadu_ps(&x[i]);
      __m256 scaled = _mm256_mul_ps(vx, scale);
      __m256i idx = _mm256_cvtps_epi32(scaled);
      __m256i idx_cos = _mm256_add_epi32(idx, quarter_pi);

      idx_cos = _mm256_and_si256(idx_cos, mask);

#if defined(__AVX2__)
      __m256 cosv = _mm256_i32gather_ps(lookup.data(), idx_cos, 4);
#else
      // fallback gather for AVX1
      float cos_tmp[VL];
      int idxc_a[VL];
      _mm256_store_si256((__m256i *)idxc_a, idx_cos);
      for (std::size_t k = 0; k < VL; ++k) {
        cos_tmp[k] = lookup[idxc_a[k]];
      }
      __m256 cosv = _mm256_load_ps(cos_tmp);
#endif
      _mm256_storeu_ps(&c[i], cosv);
    }

    // scalar remainder
    for (; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      std::size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      c[i] = lookup[idx_cos];
    }
#else
    // No AVX: scalar path
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t idx = static_cast<std::size_t>(x[i] * SCALE) & MASK;
      std::size_t idx_cos = (idx + NR_SAMPLES / 4) & MASK;
      c[i] = lookup[idx_cos];
    }
#endif
  }
};

template <std::size_t NR_SAMPLES>
LookupAVXBackend<NR_SAMPLES>::LookupAVXBackend()
    : impl(std::make_unique<Impl>()) {}

template <std::size_t NR_SAMPLES>
LookupAVXBackend<NR_SAMPLES>::~LookupAVXBackend() = default;

template <std::size_t NR_SAMPLES>
void LookupAVXBackend<NR_SAMPLES>::init(size_t) {
  impl->init();
}

template <std::size_t NR_SAMPLES>
void LookupAVXBackend<NR_SAMPLES>::compute_sinf(std::size_t n, const float *x,
                                                float *s) const {
  impl->compute_sinf(n, x, s);
}

template <std::size_t NR_SAMPLES>
void LookupAVXBackend<NR_SAMPLES>::compute_cosf(std::size_t n, const float *x,
                                                float *c) const {
  impl->compute_cosf(n, x, c);
}

template <std::size_t NR_SAMPLES>
void LookupAVXBackend<NR_SAMPLES>::compute_sincosf(std::size_t n,
                                                   const float *x, float *s,
                                                   float *c) const {
  impl->compute_sincosf(n, x, s, c);
}

// Explicit instantiations
template class LookupAVXBackend<16384>;
template class LookupAVXBackend<32768>;
