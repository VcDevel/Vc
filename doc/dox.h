/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

/*!
\mainpage
\image html logo.png

The %Vc library implements portable, zero-overhead C++ types for explicitly data-parallel
programming.

The 1.x releases ship implementations for x86 SIMD instruction sets: SSE, AVX, AVX2, and
the Xeon Phi (MIC). A scalar implementation ensures full portability to any C++11 capable
compiler and target system.

This documentation is structured in three main areas:
1. Several manually written documentation pages in the *Overview Documents*.
   They cover an introduction to SIMD and data-parallelism, portability issues, macros,
   how to set up the build system, and examples / tutorials.
   \li \subpage intro
   \li \subpage portability
   \li \subpage vcmacros
   \li \subpage featuremacros
   \li \subpage buildsystem
   \li \subpage examples

2. The *API Reference* section contains a manually structured access to the API
   documentation generated from the %Vc sources.
   \li \ref Vectors
   \li \ref Masks
   \li \ref SimdArray
   \li \ref Simdize
   \li \ref Math
   \li \ref Utilities
   \li \ref Containers

3. The *Indexes* section contains automatically generated indexes to the same API
   documentation.

\page intro Introduction

Recent generations of CPUs, and GPUs in particular, require data-parallel codes for full efficiency.
Data parallelism requires that the same sequence of operations is applied to different input data.
CPUs and GPUs can thus reduce the necessary hardware for instruction decoding and scheduling in favor of more arithmetic and logic units, which execute the same instructions synchronously.
On CPU architectures this is implemented via SIMD registers and instructions.
A single SIMD register can store N values and a single SIMD instruction can execute N operations on those values.
On GPU architectures N threads run in perfect sync, fed by a single instruction decoder/scheduler.
Each thread has local memory and a given index to calculate the offsets in memory for loads and stores.

Current C++ compilers can do automatic transformation of scalar codes to SIMD instructions (auto-vectorization).
However, the compiler must reconstruct an intrinsic property of the algorithm that was lost when the developer wrote a purely scalar implementation in C++.
Consequently, C++ compilers cannot vectorize any given code to its most efficient data-parallel variant.
Especially larger data-parallel loops, spanning over multiple functions or even translation units, will often not be transformed into efficient SIMD code.

The %Vc library provides the missing link.
Its types enable explicitly stating data-parallel operations on multiple values.
The parallelism is therefore added via the type system.
Competing approaches state the parallelism via new control structures and consequently new semantics inside the body of these control structures.

If you are new to vectorization please read this following part and make sure you
understand it:
- The term *vector* used for data-parallel programming is not about the vectors you
  studied in math classes.
- Do not confuse *vector* with containers that also go by the same name. SIMD vectors
  actually do implement some aspect of a container, but they are closer to a fixed-sized
  `std::array` than to a dynamically resizable `std::vector`.
- The *vector* type in %Vc is defined by the target hardware as a group of values with a
  fixed number of entries (\VSize{T}).
  Typically one Vc::Vector object then fits into a SIMD register on the target system.
  Such a SIMD register consequently stores \VSize{T} scalar values; in contrast to a
  general purpose register, which stores only one scalar value.
  This value \VSize{T} is thus an unchangeable property of the hardware and not a variable
  in the %Vc API.
  You can access the \VSize{T} value via the static Vc::Vector::size() function.
  Since this function is a constant expression you can also use it for template arguments.
- Note that some hardware may use different vector register widths for different data
  types.
  For example, AVX has instructions for 256-bit floating point registers, but only 128-bit
  integer instructions, which is why the integral Vc::Vector types use the SSE
  implementation for AVX target systems.

\par Example 1:

You can modify a function to use vector types and thus implement a horizontal vectorization. The
original scalar function could look like this (If you are confused about the adjective
"scalar" in this context, note that the function mathematically does a vector to vector
transformation. However, the computer computes it with \em scalar instructions, i.e. one
value per operand.):
\code
void normalize(float &x, float &y, float &z)
{
  const float d = std::sqrt(x * x + y * y + z * z);
  x /= d;
  y /= d;
  z /= d;
}
\endcode
To vectorize the \c normalize function with %Vc, the types must be substituted by their %Vc counterparts and math functions
must use the %Vc implementation (which is, per default, also imported into \c std namespace):
\code
using Vc::float_v;

void normalize(float_v &x, float_v &y, float_v &z)
{
  const float_v d = Vc::sqrt(x * x + y * y + z * z);
  x /= d;
  y /= d;
  z /= d;
}
\endcode
The latter function is able to normalize four 3D vectors when compiled for SSE in the same
time the former function normalizes one 3D vector.

For completeness, note that you can optimize the division in the normalize function further:
\code
  const float_v d_inv = float_v::One() / Vc::sqrt(x * x + y * y + z * z);
  const float_v d_inv = Vc::rsqrt(x * x + y * y + z * z); // less accurate, but faster
\endcode
Then you can multiply \c x, \c y, and \c z with \c d_inv, which is considerably faster than three
divisions.

As you can probably see, the new challenge with %Vc is the use of good data-structures which
support horizontal vectorization. Depending on your problem at hand this may become the main
focus of design (it does not have to be, though).

\section intro_alignment Alignment

\subsection intro_alignment_background What is Alignment

If you do not know what alignment is, and why it is important, read on, otherwise skip to \ref
intro_alignment_tools. Normally the alignment of data is an implementation detail left to the
compiler. Until C++11, the language did not even have any (official) means to query or modify
alignment.

Most data types require more than one Byte for storage. Thus, even most atomic data types span
several locations in memory. E.g. if you have a pointer to \c float, the address stored in this
pointer just determines the first of four Bytes of the \c float. Naively, one could think that
any address (which belongs to the process) can be used to store such a float. While this is true
for some architectures, some architectures may terminate the process when a misaligned pointer is
dereferenced. The natural alignment for atomic data types typically is the same as their size.
Thus the address of a \c float object should always be a multiple of 4 Bytes.

Alignment becomes more important for SIMD data types.
1. There are different instructions to load/store aligned and unaligned vectors. The unaligned
load/stores recently were greatly improved in x86 CPUs. Still, the rule of thumb
says that aligned loads/stores are faster.
2. Access to an unaligned vector with an instruction that expects an aligned vector crashes the
application. Once you write vectorized code you might want to make it a habit to check crashes
for unaligned addresses.
3. Memory allocation on the heap will return addresses aligned to some system specific alignment
rule. E.g. Linux 32bit aligns on 8 Bytes, while Linux 64bit aligns on 16 Bytes. Both alignments
are not strict enough for AVX vectors. Worse, if you develop on Linux 64bit with SSE you won't
notice any problems until you switch to a 32bit build or AVX.
4. Placement on the stack is determined at compile time and requires the compiler to know the
alignment restrictions of the type.
5. The size of a cache line is just two or four times larger than the SIMD types (if not equal).
Thus, if you load several vectors consecutively from memory every fourth, second, or even every
load will have to be read from two different cache lines. This is called a cache line split. They
lead to degraded performance, which becomes very noticeable for memory intensive code.

\subsection intro_alignment_tools Tools

%Vc provides several classes and functions to get alignment right.
\li Vc::VectorAlignment is a compile time constant that equals the largest alignment restriction
                  (in Bytes) for the selected target architecture.
\li Vc::AlignedBase, Vc::VectorAlignedBase, and Vc::MemoryAlignedBase implement the alignment
                  restrictions needed for aligned vector loads and stores. They set the
                  alignment attribute and reimplement the \c new and \c delete operators,
                  returning correctly aligned pointers to the heap.
\li Vc::malloc and Vc::free are meant as replacements for \c malloc and \c free. They can be used
                  to allocate any type of memory with an abstract alignment restriction: \ref
                  Vc::MallocAlignment. Note, that (like \c malloc) the memory is only allocated
                  and not initialized. If you allocate memory for a type that has a constructor,
                  use the placement new syntax to initialize the memory.
\li Vc::Allocator is an STL compatible allocator class that behaves as specified in the C++
                  specification, implementing the optional support for over-aligned types.
                  Therefore, memory addresses returned from this allocator will always be
                  aligned to at least the constraints attached to the type \c T. STL containers
                  will already default to Vc::Allocator for Vc::Vector<T>. For all other
                  composite types you want to use, you can take the \ref Vc_DECLARE_ALLOCATOR
                  convenience macro to set is as default.
\li Vc::Memory, Vc::Memory<V, Size, 0u>, Vc::Memory<V, 0u, 0u>
                  The three different variants of the memory class can be used like a more
                  convenient C-array. It supports two-dimensional statically sized arrays and
                  one-dimensional statically and dynamically sized arrays. The memory can be
                  accessed easily via aligned vectors, but also via unaligned vectors or
                  gathers/scatters.



\page portability Portability Issues

One of the major goals of %Vc is to ease development of portable code, while achieving highest
possible performance that requires target architecture specific instructions. This is possible
through having just a single type use different implementations of the same API depending on the
target architecture. Many of the details of the target architecture are often dependent on the
compiler flags that were used. Also there can be subtle differences between the implementations
that could lead to problems. This page aims to document all issues you might need to know about.

\par Compiler Flags

\li \e GCC: The compiler should be called with the `-march=\<target\>` flag. Take a look at the GCC
manpage to find all possibilities for `\<target\>`.
If no SIMD instructions are enabled via compiler flags, %Vc must fall back
to the scalar implementation.
\li \e Clang: The same as for GCC applies.
\li \e ICC: The same as for GCC applies (at least on Linux).
\li \e MSVC: The compiler supports, among others, the `/arch:AVX`, `/arch:AVX2` and `/arch:AVX512 flags.
Without such a flag, at least SSE2 is enabled.

\par Where does the final executable run?

You must be aware of the fact that a binary that is built for a given SIMD hardware may not run
on a processor that does not have these instructions. The executable will work fine as long as no
such instruction is actually executed and only crash at the place where such an instruction is
used. Thus it is better to check at application start whether the compiled in SIMD hardware is
really supported on the executing CPU. This can be determined with the
currentImplementationSupported function.

If you want to distribute a binary that runs correctly on many different systems you either must
restrict it to the least common denominator (which often is SSE2), or you must compile the code
several times, with the different target architecture compiler options. A simple way to combine
the resulting executables would be via a wrapping script/executable that determines the correct
executable to use. A more sophisticated option is the use of the ifunc attribute GCC provides.
Other compilers might provide similar functionality.

\par Guarantees

It is guaranteed that:
\li \code int_v::Size == uint_v::Size == float_v::Size \endcode
\li \code short_v::Size == ushort_v::Size \endcode

\par Important Differences between Implementations

\li Obviously the number of entries in a vector depends on the target architecture.
\li Hardware that does not support 16-Bit integer vectors can implement the short_v and ushort_v
API via 32-Bit integer vectors. Thus, some of the overflow behavior might be slightly different,
and truncation will only happen when the vector is stored to memory.

\section portability_compilerquirks Compiler Quirks

Since SIMD is not part of the C/C++ language standards %Vc abstracts more or less standardized
compiler extensions. Sadly, not every issue can be transparently abstracted.
Therefore this will be the place where differences are documented:
\li MSVC is incapable of parameter passing by value, if the type has alignment restrictions. The
consequence is that all %Vc vector types and any type derived from Vc::VectorAlignedBase cannot be
used as function parameters, unless a pointer is used (this includes reference and
const-reference). So \code
void foo(Vc::float_v) {}\endcode does not compile, while \code
void foo(Vc::float_v &) {}
void foo(const Vc::float_v &) {}
void foo(Vc::float_v *) {}
\endcode all work.
Normally you should prefer passing by value since a sane compiler will then pass the data in a
register and does not have to store/load the data to/from the stack. %Vc defines \c
Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN for such cases. Also the %Vc vector types contain a composite
typedef \c AsArg which resolves to either const-ref or const-by-value. Thus, you can always use
\code void foo(Vc::float_v::AsArg) {}\endcode.


\page vcmacros Pre-defined Macros

The %Vc library defines a few macros that you may rely on in your code:

\section vc_impl Implementation Identification

One or more of the following macros will be defined:
\li \ref Vc_IMPL_Scalar
\li \ref Vc_IMPL_SSE
\li \ref Vc_IMPL_SSE2
\li \ref Vc_IMPL_SSE3
\li \ref Vc_IMPL_SSSE3
\li \ref Vc_IMPL_SSE4_1
\li \ref Vc_IMPL_SSE4_2
\li \ref Vc_IMPL_AVX
\li \ref Vc_IMPL_AVX2

You can use these macros to enable target-specific implementations.
In general, it is better to rely on function overloading or template mechanisms, though.
Per default, code compiled against the %Vc headers will use the instruction set that the compiler advertises as available.
For example, compiling with "g++ -mssse3" chooses the SSE implementation (Vc::VectorAbi::Sse) with instructions from SSE, SSE2, SSE3 and SSSE3.
After you include a %Vc header, you will have the following macros available, which you can (but normally should not) use to determine the implementation %Vc uses:

\section vc_size Vector/Mask Sizes

The macros \ref Vc_DOUBLE_V_SIZE, \ref Vc_FLOAT_V_SIZE, \ref Vc_INT_V_SIZE, \ref Vc_UINT_V_SIZE, \ref Vc_SHORT_V_SIZE, and \ref Vc_USHORT_V_SIZE make the default vector width accessible in the preprocessor.
In most cases you should prefer the Vector::size() function, though.
Since this function is \c constexpr you can use it for compile-time decisions (e.g. as template argument).

\section vc_compiler Compiler Identification (and related)

- \ref Vc_GCC
- \ref Vc_CLANG
- \ref Vc_APPLECLANG
- \ref Vc_ICC
- \ref Vc_MSVC
- \ref Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN

\section vc_version Version Macros

- \ref Vc_VERSION_STRING
- \ref Vc_VERSION_NUMBER
- \ref Vc_VERSION_CHECK

\section vc_boilerplate Boilerplate Code Generation

- \ref Vc_SIMDIZE_INTERFACE


\page featuremacros Feature Macros

You can define the following macros to enable/disable specific features:

\section set_vc_impl Vc_IMPL

If you want to force compilation against a specific implementation of the vector classes you can set the macro Vc_IMPL to either
\c Scalar, \c SSE, \c SSE2, \c SSE3, \c SSSE3, \c SSE4_1, \c SSE4_2, \c AVX, \c AVX2, or \c MIC.
Additionally, you may (should) append \c +XOP, \c +FMA4, \c +FMA, \c +SSE4a, \c +F16C, \c +BMI2, and/or \c +POPCNT.
For example, `-D Vc_IMPL=SSE+XOP+FMA4` tells the Vc library to use the best SSE instructions available for the target (according to the information provided by the compiler) and additionally use XOP and FMA4 instructions (this might be a good choice for some AMD processors, which support AVX but may perform slightly better if only SSE widths are used).
Setting \c Vc_IMPL to \c SSE forces the SSE instruction set, but lets the headers figure out the exact SSE revision to use, or, if that fails, uses SSE4.1.

If you do not specify \c Vc_IMPL the %Vc headers determine the implementation from compiler-specific pre-defined macros (which in turn are determined from compiler flags that determine the target micro-architecture, such as \c -mavx2).

\section Vc_NO_STD_FUNCTIONS

If this macro is defined, the %Vc math functions are not imported into the \c std namespace.
They are still available in the %Vc namespace and through [ADL](http://en.cppreference.com/w/cpp/language/adl).

\section Vc_NO_VERSION_CHECK

Define this macro to disable the safety check for the libVc version.
The check generates a small check for every object file, which is called at startup, i.e. before
the main function.

\section Vc_CHECK_ALIGNMENT

If this macro is defined %Vc will assert correct alignment for all objects that require correct alignment.
This can be very useful to debug crashes resulting from misaligned memory accesses.
This check will introduce a significant overhead.

\section Vc_ENABLE_FLOAT_BIT_OPERATORS

Define this macro to enable bitwise operators (&, |, ^, ~) on floating-point vectors. Since these
operators are not provided for the builtin floating-point types, the default is to not provide
them for SIMD vector types as well.



\page buildsystem Build System

%Vc uses CMake as its buildsystem.
It also provides much of the CMake logic it uses for itself for other projects that use CMake and %Vc.
Here's an (incomplete) list of features you can get from the CMake scripts provided with %Vc:
\li check for a required %Vc version
\li locate libVc and %Vc includes
\li compiler flags to work around %Vc related quirks/bugs in specific compilers
\li compiler flags to enable/disable SIMD instruction sets (defaults to full support for the host system)

\section buildsystem_variables CMake Variables

To make use of these features simply copy the FindVc.cmake as installed by %Vc to your project.
Add \code find_package(Vc [version] [REQUIRED]) \endcode to your CMakeLists.txt. After that you
can use the following variables:
\li \e Vc_FOUND: tells whether the package was found
\li \e Vc_INCLUDE_DIR: you must add this to your include directories for the targets that you
want to compile against %Vc: \code include_directories(${Vc_INCLUDE_DIR}) \endcode
\li \e Vc_DEFINITIONS: recommended preprocessor definitions. You can use them via \c add_definitions.
\li \e Vc_COMPILE_FLAGS: recommended compiler flags. You can use them via the
\li \e Vc_ARCHITECTURE_FLAGS: recommended compiler flags for a selected target
microarchitecture. You can use them via the \c COMPILE_OPTIONS property or via \c
add_compile_options.
\li \e Vc_ALL_FLAGS: a list combining the above three variables. Use it to conveniently
set all required compiler flags in one place (e.g. via \c add_compile_options).

The following variables might be of interest, too:
\li \e Vc_SSE_INTRINSICS_BROKEN
\li \e Vc_AVX_INTRINSICS_BROKEN
\li \e Vc_XOP_INTRINSICS_BROKEN
\li \e Vc_FMA4_INTRINSICS_BROKEN

\section buildsystem_macros CMake Macros

The macro vc_compile_for_all_implementations is provided to help with compiling a given source
file multiple times with all different possible SIMD targets for the given architecture.
Example:
\verbatim
vc_compile_for_all_implementations(objs src/trigonometric.cpp
  FLAGS -DSOME_FLAG
  EXCLUDE Scalar SSE2)
\endverbatim
You can specify an arbitrary number of additional compiler flags after the FLAGS argument. These
flags will be used for all compiler calls. After an optional EXCLUDE argument you can specify targets
that you want to exclude. After an optional ONLY argument you can specify targets that you want
to compile for. (So either you exclude some, or you explicitly list the targets you want.)

Often it suffices to have SSE2 or SSE3 as the least common denominator and provide SSE4_1 and
AVX. Here is the currently complete list of possible targets the macro will compile for:
\li Scalar
\li SSE2
\li SSE3
\li SSSE3
\li SSE4_1
\li SSE4_2
\li SSE3+SSE4a
\li SSE+XOP+FMA4
\li AVX
\li AVX+XOP+FMA4
\li AVX2+FMA+BMI2

\section buildsystem_other Using Vc without CMake

If your project does not use CMake all you need to do is the following:
\li Find the header file "Vc/Vc" and add its path to your include paths.
\li Find the library libVc and link to it.
\li Ensure you use the right compiler flags to enable the relevant SIMD instructions.


\defgroup Vectors Vectors
\defgroup Masks Masks
\defgroup SimdArray SIMD Array
\defgroup Simdize simdize<T>
\defgroup Math Math
\defgroup Utilities Utilities
\defgroup Containers Containers

\addtogroup Vectors

The vector classes abstract the SIMD registers and their according instructions into types that
feel very familiar to C++ developers.

Note that the documented types Vc::float_v, Vc::double_v, Vc::int_v, Vc::uint_v, Vc::short_v,
and Vc::ushort_v are actually \c typedefs of the \c Vc::Vector<T> class:
\code
namespace Vc {
  template<typename T> class Vector;
  typedef Vector<double> double_v;
  typedef Vector<float> float_v;
  // ...
}
\endcode

\par Some general information on using the vector classes:

Generally you can always mix scalar values with vectors as %Vc will automatically broadcast the
scalar to a vector and then execute a vector operation. But, in order to ensure that implicit
type conversions only happen as defined by the C standard, there is only a very strict implicit
scalar to vector constructor:
\code
int_v a = 1;     // good:             int_v(int)
uint_v b = 1u;   // good:             uint_v(unsigned int)
uint_v c = 1;    // does not compile: uint_v(int)
float_v d = 1;   // does not compile: float_v(int)
float_v e = 1.;  // does not compile: float_v(double)
float_v f = 1.f; // good:             float_v(float)
\endcode

The following ways of initializing a vector are not allowed:
\code
int_v v(3, 2, 8, 0); // constructor does not exist because it is not portable
int_v v;
v[0] = 3; v[1] = 2; v[2] = 8; v[3] = 0; // do not hardcode the number of entries!
// You can not know whether somebody will compile with %Vc Scalar where int_v::Size == 1
\endcode

Instead, if really necessary you can do:
\code
Vc::int_v v;
for (int i = 0; i < int_v::Size; ++i) {
  v[i] = f(i);
}
// which is equivalent to:
v.fill(f);
// or:
v = int_v::IndexesFromZero().apply(f);
\endcode



\addtogroup Masks

Mask classes are abstractions for the results of vector comparisons. The actual implementation
differs depending on the SIMD instruction set. On SSE they contain a full 128-bit datatype while
on a different architecture they might be bit-fields.


\addtogroup Utilities

Additional classes, macros, and functions that help to work more easily with the main vector
types.


\addtogroup Containers

For some problems, standard (or third-party) containers can be used.
Simply use a `value_type` of `Vc::Vector<T>`.
However, this requires:
\li You actually have control over the data structures and can design/modify them for easy
    vectorization usage.
\li The access patterns are non-random. Because random access to individual `value_type`
    \em elements is going to be a pain (two subscripts, first into the container, then
    into the `Vc::Vector`)

Therefore, for some problems you need to work with containers over elements of
non-`Vector` type (e.g. of type `double` or a `struct`).
Vc provides some help:
\li Vc::vector
\li Vc::array
\li Vc::span
\li Vc::Common::Memory (discouraged)
\li Vc::Common::InterleavedMemoryWrapper


\addtogroup Math

Functions that implement math functions. Take care that some of the implementations will return
results with less precision than what the FPU calculates.


\addtogroup SimdArray

This set of class templates and associated functions and operators enables
data-parallel algorithms and data structures requiring a user-defined number of elements
(fixed at compile time, in contrast to \c std::valarray where the number of elements is
 only determined at run time).
The main motivation for a user-defined number of elements is the need for type conversion
and thus a guaranteed equal number of elements in data-parallel vectors for e.g. \c float
and \c int.
A typical pattern looks like this:
\code
using floatv = Vc::float_v;
using doublev = Vc::SimdArray<double, floatv::size()>;
using intv = Vc::SimdArray<int, floatv::size()>;
using uintv = Vc::SimdArray<unsigned int, floatv::size()>;
\endcode

The second motivation for a user-defined number of elements is that many vertical
vectorizations require a fixed number of elements (i.e. number known at development time
and not chosen at compile time).
The implementation can then choose how to support this number most efficiently with the
available hardware resources.
Consider, for example, a need for processing 12 values in parallel.
On x86 with AVX, the implementation could build such a type from one AVX and one SSE register.

In contrast to \c std::array the types behave like the Vc::Vector types, implementing the same
operators and functions.
The semantics with regard to implicit conversions differ slightly:
The Vc::Vector conversion rules are safer with regard to source compatibility.
The Vc::SimdArray conversion rules are less strict and could potentially lead to portability issues.
Therefore, it is best to stick to the pattern of type aliases shown above.
*/

/**
 * \brief Vector Classes Namespace
 *
 * All functions and types of %Vc are defined inside the %Vc namespace.
 *
 * \internal
 * Internal types and functions should be defined either in the Vc::Detail namespace or in
 * a `Vc::<Impl>` namespace such as Vc::SSE.
 */
namespace Vc
{
#define INDEX_TYPE Vc::SimdArray<int, size()>
#define VECTOR_TYPE Vc::Vector<T>
#define ENTRY_TYPE T
#define MASK_TYPE Vc::Mask<T>
#define EXPONENT_TYPE Vc::SimdArray<int, size()>

#include "dox-math.h"

#undef INDEX_TYPE
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef EXPONENT_TYPE

/**
 * \name SIMD Support Feature Macros
 * \ingroup Utilities
 */
//@{
///\addtogroup Utilities
//@{
/**
 * This macro is defined if the current translation unit is compiled with XOP instruction support.
 */
#define Vc_IMPL_XOP
/**
 * This macro is defined if the current translation unit is compiled with FMA4 instruction support.
 */
#define Vc_IMPL_FMA4
/**
 * This macro is defined if the current translation unit is compiled with F16C instruction support.
 */
#define Vc_IMPL_F16C
/**
 * This macro is defined if the current translation unit is compiled with POPCNT instruction support.
 */
#define Vc_IMPL_POPCNT
/**
 * This macro is defined if the current translation unit is compiled with SSE4a instruction support.
 */
#define Vc_IMPL_SSE4a
/**
 * This macro is defined if the current translation unit is compiled without any SIMD support.
 */
#define Vc_IMPL_Scalar
/**
 * This macro is defined if the current translation unit is compiled with any version of SSE (but not
 * AVX).
 */
#define Vc_IMPL_SSE
/**
 * This macro is defined if the current translation unit is compiled with SSE2 instruction support
 * (excluding SSE3 and up).
 */
#define Vc_IMPL_SSE2
/**
 * This macro is defined if the current translation unit is compiled with SSE3 instruction support (excluding SSSE3 and up).
 */
#define Vc_IMPL_SSE3
/**
 * This macro is defined if the current translation unit is compiled with SSSE3 instruction support (excluding SSE4.1 and up).
 */
#define Vc_IMPL_SSSE3
/**
 * This macro is defined if the current translation unit is compiled with SSE4.1 instruction support (excluding SSE4.2 and up).
 */
#define Vc_IMPL_SSE4_1
/**
 * This macro is defined if the current translation unit is compiled with SSE4.2 instruction support (excluding AVX and up).
 */
#define Vc_IMPL_SSE4_2
/**
 * This macro is defined if the current translation unit is compiled with AVX instruction support (excluding AVX2 and up).
 */
#define Vc_IMPL_AVX
/**
 * This macro is defined if the current translation unit is compiled with AVX2 instruction support.
 */
#define Vc_IMPL_AVX2
//@}
//@}

/**
 * \name SIMD Vector Size Macros
 * \ingroup Utilities
 */
//@{
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a double_v.
 */
#define Vc_DOUBLE_V_SIZE
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a float_v.
 */
#define Vc_FLOAT_V_SIZE
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a int_v.
 */
#define Vc_INT_V_SIZE
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a uint_v.
 */
#define Vc_UINT_V_SIZE
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a short_v.
 */
#define Vc_SHORT_V_SIZE
/**
 * \ingroup Utilities
 * An integer (for use with the preprocessor) that gives the number of entries in a ushort_v.
 */
#define Vc_USHORT_V_SIZE
//@}

} // namespace Vc

// vim: ft=doxygen
