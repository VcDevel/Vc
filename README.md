# Vc: portable, zero-overhead C++ types for explicitly data-parallel programming

Recent generations of CPUs, and GPUs in particular, require data-parallel codes
for full efficiency. Data parallelism requires that the same sequence of
operations is applied to different input data. CPUs and GPUs can thus reduce
the necessary hardware for instruction decoding and scheduling in favor of more
arithmetic and logic units, which execute the same instructions synchronously.
On CPU architectures this is implemented via SIMD registers and instructions.
A single SIMD register can store N values and a single SIMD instruction can
execute N operations on those values. On GPU architectures N threads run in
perfect sync, fed by a single instruction decoder/scheduler. Each thread has
local memory and a given index to calculate the offsets in memory for loads and
stores.

Current C++ compilers can do automatic transformation of scalar codes to SIMD
instructions (auto-vectorization). However, the compiler must reconstruct an
intrinsic property of the algorithm that was lost when the developer wrote a
purely scalar implementation in C++. Consequently, C++ compilers cannot
vectorize any given code to its most efficient data-parallel variant.
Especially larger data-parallel loops, spanning over multiple functions or even
translation units, will often not be transformed into efficient SIMD code.

The Vc library provides the missing link. Its types enable explicitly stating
data-parallel operations on multiple values. The parallelism is therefore added
via the type system. Competing approaches state the parallelism via new control
structures and consequently new semantics inside the body of these control
structures.

Vc is a free software library to ease explicit vectorization of C++ code. It
has an intuitive API and provides portability between different compilers and
compiler versions as well as portability between different vector instruction
sets. Thus an application written with Vc can be compiled for:

* AVX and AVX2
* SSE2 up to SSE4.2 or SSE4a
* Scalar (fallback which works everywhere)
* MIC (for Vc 1.0)
* NEON (in development)
* NVIDIA GPUs / CUDA (in development)


## Build Requirements

cmake >= 3.0

C++11 Compiler:

* GCC >= 4.8.1
* clang >= 3.4
* ICC >= 15.0.3
* Visual Studio (not ready for Vc 1.0 yet)


## Building and Installing Vc

* Create a build directory:

```sh
$ mkdir build
$ cd build
```

* Call cmake with the relevant options:

```sh
$ cmake -DCMAKE_INSTALL_PREFIX=/opt/Vc -DBUILD_TESTING=OFF <srcdir>
```

* Build and install:

```sh
$ make -j16
$ make install
```

## Documentation

The documentation is generated via [doxygen](http://doxygen.org). You can build
the documentation by running `doxygen` in the `doc` subdirectory.
Alternatively, you can find nightly builds of the documentation at:

* [master branch](https://web-docs.gsi.de/~mkretz/Vc-master/)
* [1.0 branch](https://web-docs.gsi.de/~mkretz/Vc-1.0/)
* [0.7 branch](https://web-docs.gsi.de/~mkretz/Vc-0.7/)

## Publications

* [M. Kretz, "Extending C++ for Explicit Data-Parallel Programming via SIMD
  Vector Types", Goethe University Frankfurt, Dissertation,
  2015.](http://publikationen.ub.uni-frankfurt.de/frontdoor/index/index/docId/38415)
* [M. Kretz and V. Lindenstruth, "Vc: A C++ library for explicit
  vectorization", Software: Practice and Experience,
  2011.](http://dx.doi.org/10.1002/spe.1149)
* [M. Kretz, "Efficient Use of Multi- and Many-Core Systems with Vectorization
  and Multithreading", University of Heidelberg,
  2009.](http://code.compeng.uni-frankfurt.de/attachments/13/Diplomarbeit.pdf)


## Communication

A channel on the freenode IRC network is reserved for discussions on Vc:
[##vc on freenode](irc://chat.freenode.net:6665/##vc)
([via SSL](ircs://chat.freenode.net:7000/##vc))

There exist two mailinglists:

* [List for users of Vc, i.e. developers that use
  Vc](https://compeng.uni-frankfurt.de/mailman/listinfo/vc)
* [List to discuss the development of Vc
  itself](https://compeng.uni-frankfurt.de/mailman/listinfo/vc-devel)

Feel free to use the GitHub issue tracker for questions, too.

## License

Vc up to version 0.7.4 was released under the [LGPL 3 license](http://opensource.org/licenses/LGPL-3.0).
Since then the code was relicensed to the [3-clause BSD license](http://opensource.org/licenses/BSD-3-Clause) and subsequent releases will use the BSD.
