# `std::experimental::simd`
portable, zero-overhead C++ types for explicitly data-parallel programming

This package implements ISO/IEC TS 19570:2018 Section 9 "Data-Parallel Types".
It is targetting inclusion into libstdc++. By default, the `install.sh` script
places the `std::experimental::simd` headers into the directory where the
standard library of your C++ compiler (identified via `$CXX`) resides.

The implementation derives from https://github.com/VcDevel/Vc.
It is only tested and supported with GCC 9, even though it may (partially) work
with older GCC versions.

## Target support

* x86_64 is the main development platform and thoroughly tested. This includes
  support from SSE-only up to AVX512 on Xeon Phi or Xeon CPUs.
* aarch64 was tested and verified to work. No significant performance evaluation
  was done.
* ARM NEON in general should work, too.
* IBM Power support received minimal testing.
* In any case, a fallback to correct execution via builtin arthmetic types is
  available for all targets.

## Examples

### Scalar Product

Let's start from the code for calculating a 3D scalar product using builtin floats:
```cpp
using Vec3D = std::array<float, 3>;
float scalar_product(Vec3D a, Vec3D b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
```

Using `simd`, we can easily vectorize the code using the `native_simd<float>` type:
```cpp
using std::experimental::native_simd;
using Vec3D = std::array<native_simd<float>, 3>;
native_simd<float> scalar_product(Vec3D a, Vec3D b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
```

The above will scale to 1, 4, 8, 16, etc. scalar products calculated in parallel, depending
on the target hardware's capabilities.

For comparison, the same vectorization using Intel SSE intrinsics is more verbose, uses
prefix notation (i.e. function calls), and neither scales to AVX or AVX512, nor is it
portable to different SIMD ISAs:
```cpp
using Vec3D = std::array<__m128, 3>;
__m128 scalar_product(Vec3D a, Vec3D b) {
  return _mm_add_ps(_mm_add_ps(_mm_mul_ps(a[0], b[0]), _mm_mul_ps(a[1], b[1])),
                    _mm_mul_ps(a[2], b[2]));
}
```

## Install Instructions

```sh
$ ./install.sh
```

Use `--help` to learn about the available options.

## Build Requirements

none. It's header-only.

However, to build the unit tests you will need:
* cmake >= 3.0
* GCC >= 9.1

To execute all AVX512 unit tests, you will need the Intel SDE.

## Building the tests

```sh
$ make test
```

This will create a build directory, run cmake, compile the tests, and execute the tests.

## Documentation

https://en.cppreference.com/w/cpp/experimental/simd

## Publications

* [M. Kretz, "Extending C++ for Explicit Data-Parallel Programming via SIMD
  Vector Types", Goethe University Frankfurt, Dissertation,
  2015.](http://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/38415)
* [M. Kretz and V. Lindenstruth, "Vc: A C++ library for explicit
  vectorization", Software: Practice and Experience,
  2011.](http://dx.doi.org/10.1002/spe.1149)
* [J. Hoberock, "Working Draft, C++ Extensions for Parallelism Version 2", 2019](https://wg21.link/N4808)

## Communication

A channel on the freenode IRC network is reserved for discussions on Vc:
[##vc on freenode](irc://chat.freenode.net:6667/##vc)
([via SSL](ircs://chat.freenode.net:6697/##vc))

Feel free to use the GitHub issue tracker for questions.
Alternatively, there's a [mailinglist for users of
Vc](https://compeng.uni-frankfurt.de/mailman/listinfo/vc)

## License

The `simd` headers, tests, and benchmarks are released under the terms of the
[3-clause BSD license](http://opensource.org/licenses/BSD-3-Clause).
