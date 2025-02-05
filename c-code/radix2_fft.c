#include "fft_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void swap(Complex *a, Complex *b) {
  Complex temp = *a;
  *a = *b;
  *b = temp;
}

void radix2_fft(Complex *x, size_t n, int inverse) {

  int j = 0;
  for (int i = 1; i < n; i++) {
    size_t bit = n >> 1;
    while (j >= bit) {
      j -= bit;
      bit >>= 1;
    }
    j += bit;
    if (i < j)
      swap(&x[i], &x[j]);
  }

  for (int m = 2; m <= n; m *= 2) {
    double angle = 2 * M_PI / m * (inverse ? 1 : -1);
    Complex wm;
    wm.real = cos(angle);
    wm.imag = sin(angle);

    // Loop over each sub-DFT.
    for (j = 0; j < n; j += m) {
      Complex w;
      w.real = 1.0;
      w.imag = 0.0;
      for (int k = 0; k < m / 2; k++) {
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
    for (int i = 0; i < n; i++) {
      x[i].real /= n;
      x[i].imag /= n;
    }
  }
}

void radix2_forward(Complex *data, size_t n) { radix2_fft(data, n, 0); }

void radix2_inverse(Complex *data, size_t n) { radix2_fft(data, n, 1); }

FFTInterface FFT_Radix2 = {.fft_forward = radix2_forward,
                           .fft_inverse = radix2_inverse};

void print_complex_array(const char *label, Complex *data, size_t n) {
  printf("%s:\n", label);
  for (size_t i = 0; i < n; i++) {
    printf("  [%zu]: %.3f + %.3fi\n", i, data[i].real, data[i].imag);
  }
  printf("\n");
}

int main(void) {
  size_t n = 8;

  Complex input[8] = {{1, 0}, {2, 0}, {3, 0}, {4, 0},
                      {5, 0}, {6, 0}, {7, 0}, {8, 0}};

  print_complex_array("Original Input", input, n);

  FFTInterface *fftImpl = &FFT_Radix2;

  fftImpl->fft_forward(input, n);
  print_complex_array("After Forward FFT", input, n);

  fftImpl->fft_inverse(input, n);
  print_complex_array("After Inverse FFT (Recovered Data)", input, n);

  return 0;
}
