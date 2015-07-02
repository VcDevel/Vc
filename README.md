# Vc: portable, zero-overhead SIMD library for C++

The use of SIMD is becoming increasingly important with modern CPUs. The SIMD
instruction sets are being improved: new instructions are added as well as
performance improvements relative to the scalar instructions. The next
generations of CPUs will double the vector width. Neglecting SIMD in
high-performance code thus becomes more expensive, compared to the theoretical
performance of CPUs.

The use of SIMD instructions is not easy. C/C++ compilers support some
extensions to ease development for SSE and AVX. Commonly intrinsics are the
available extension of choice. Intrinsics basically map every SIMD instruction
to a C function. The use of these intrinsics leads to code which is hard to read
and maintain in addition to making portability to other vector units
complicated.

Vc is a free software library to ease explicit vectorization of C++ code. It has
an intuitive API and provides portability between different compilers and
compiler versions as well as portability between different vector instruction
sets. Thus an application written with Vc can be compiled for

* AVX
* SSE2 up to SSE4.2 or SSE4a
* Scalar (fallback which works everywhere)
* MIC (for Vc 1.0)
* NEON (in development)


## Build Requirements

cmake >= 2.8.3

C++11 Compiler:

* GCC >= 4.6
* clang >= 3.2
* ICC >= 13
* Visual Studio >= 2012


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


## Publications

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


## License

Vc is released under the LGPL 3. Since Vc is a template library this gives you
a lot of freedom. For the details you can take a look at the [Licensing FAQ of
Eigen](http://eigen.tuxfamily.org/index.php?title=Licensing_FAQ) which is a C++
template library released under the LGPL 3 or GPL 2.
