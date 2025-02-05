#ifndef FFT_INTERFACE_H
#define FFT_INTERFACE_H

#include <stddef.h>

typedef struct {
  double real;
  double imag;
} Complex;

typedef struct {
  void (*fft_forward)(Complex *data, size_t n);

  void (*fft_inverse)(Complex *data, size_t n);

} FFTInterface;

#endif
