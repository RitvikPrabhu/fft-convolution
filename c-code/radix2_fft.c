#include "fft_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void swap(Complex *a, Complex *b) {
  Complex temp = *a;
  *a = *b;
  *b = temp;
}

static void radix2_fft(Complex *x, size_t n, int inverse) {
  size_t i, j, k, m;

  j = 0;
  for (i = 1; i < n; i++) {
    size_t bit = n >> 1;
    while (j >= bit) {
      j -= bit;
      bit >>= 1;
    }
    j += bit;
    if (i < j)
      swap(&x[i], &x[j]);
  }

  for (m = 2; m <= n; m <<= 1) {
    double angle = 2 * M_PI / m * (inverse ? 1 : -1);
    Complex wm;
    wm.real = cos(angle);
    wm.imag = sin(angle);

    // Loop over each sub-DFT.
    for (j = 0; j < n; j += m) {
      Complex w;
      w.real = 1.0;
      w.imag = 0.0;
      for (k = 0; k < m / 2; k++) {
        Complex t, u;
        // t = w * x[j + k + m/2]
        t.real =
            w.real * x[j + k + m / 2].real - w.imag * x[j + k + m / 2].imag;
        t.imag =
            w.real * x[j + k + m / 2].imag + w.imag * x[j + k + m / 2].real;
        u = x[j + k];

        // Butterfly operations:
        x[j + k].real = u.real + t.real;
        x[j + k].imag = u.imag + t.imag;
        x[j + k + m / 2].real = u.real - t.real;
        x[j + k + m / 2].imag = u.imag - t.imag;

        // Update twiddle factor: w = w * wm.
        double temp_w_real = w.real * wm.real - w.imag * wm.imag;
        w.imag = w.real * wm.imag + w.imag * wm.real;
        w.real = temp_w_real;
      }
    }
  }

  if (inverse) {
    for (i = 0; i < n; i++) {
      x[i].real /= n;
      x[i].imag /= n;
    }
  }
}

static void radix2_forward(Complex *data, size_t n) { radix2_fft(data, n, 0); }

static void radix2_inverse(Complex *data, size_t n) { radix2_fft(data, n, 1); }

FFTInterface FFT_Radix2 = {.fft_forward = radix2_forward,
                           .fft_inverse = radix2_inverse};

int main(void) {
  // Define FFT size (must be a power of 2).
  size_t n = 8;

  Complex input[8] = {{1, 0}, {2, 0}, {3, 0}, {4, 0},
                      {0, 0}, {0, 0}, {0, 0}, {0, 0}};

  FFTInterface *fftImpl = &FFT_Radix2;

  return 0;
}
