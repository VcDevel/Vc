/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_GLOBAL_H
#define VC_GLOBAL_H

#define SSE    9875294
#define SSE2   9875295
#define SSE3   9875296
#define SSSE3  9875297
#define SSE4_1 9875298
#define Scalar 9875299
#define LRBni  9875300
#define SSE4_2 9875301
#define SSE4a  9875302

#ifndef VC_IMPL

#  if defined(__SSE4a__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSE4a 1
#  endif
#  if defined(__SSE4_2__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSE4_2 1
#  endif
#  if defined(__SSE4_1__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSE4_1 1
#  endif
#  if defined(__SSE3__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSE3 1
#  endif
#  if defined(__SSSE3__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSSE3 1
#  endif
#  if defined(__SSE2__)
#    define VC_IMPL_SSE 1
#    define VC_IMPL_SSE2 1
#  endif

#  if defined(VC_IMPL_SSE)
     // nothing
#  elif defined(__LRB__)
#    define VC_IMPL_LRBni 1
#  else
#    define VC_IMPL_Scalar 1
#  endif

#else // VC_IMPL

#  if VC_IMPL == Scalar
#    define VC_IMPL_Scalar 1
#  elif VC_IMPL == LRBni
#    define VC_IMPL_LRBni 1
#  elif VC_IMPL == SSE4a
#    define VC_IMPL_SSE4a 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSE4_2
#    define VC_IMPL_SSE4_2 1
#    define VC_IMPL_SSE4_1 1
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSE4_1
#    define VC_IMPL_SSE4_1 1
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSSE3
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSE3
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSE2
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif VC_IMPL == SSE
#    define VC_IMPL_SSE 1
#    if defined(__SSE4a__)
#      define VC_IMPL_SSE4a 1
#    endif
#    if defined(__SSE4_2__)
#      define VC_IMPL_SSE4_2 1
#    endif
#    if defined(__SSE4_1__)
#      define VC_IMPL_SSE4_1 1
#    endif
#    if defined(__SSE3__)
#      define VC_IMPL_SSE3 1
#    endif
#    if defined(__SSSE3__)
#      define VC_IMPL_SSSE3 1
#    endif
#    if defined(__SSE2__)
#      define VC_IMPL_SSE2 1
#    endif
#  endif
#  undef VC_IMPL

#endif // VC_IMPL

#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3)) && !defined(VC_IMPL_Scalar)
#  ifndef VC_DONT_WARN_OLD_GCC
#    warning "GCC < 4.3 does not have full support for SSE2 intrinsics. Using scalar types/operations only. Define VC_DONT_WARN_OLD_GCC to silence this warning."
#  endif
#  undef VC_IMPL_SSE
#  undef VC_IMPL_SSE2
#  undef VC_IMPL_SSE3
#  undef VC_IMPL_SSE4a
#  undef VC_IMPL_SSE4_1
#  undef VC_IMPL_SSE4_2
#  undef VC_IMPL_SSSE3
#  undef VC_IMPL_AVX
#  undef VC_IMPL_LRBni
#  define VC_IMPL_Scalar 1
#endif

# if !defined(VC_IMPL_LRBni) && !defined(VC_IMPL_Scalar) && !defined(VC_IMPL_SSE)
#  error "No suitable Vc implementation was selected! Probably VC_IMPL was set to an invalid value."
# elif defined(VC_IMPL_SSE) && !defined(VC_IMPL_SSE2)
#  error "SSE requested but no SSE2 support. Vc needs at least SSE2!"
# endif

#undef SSE
#undef SSE2
#undef SSE3
#undef SSSE3
#undef SSE4_1
#undef SSE4_2
#undef SSE4a
#undef Scalar
#undef LRBni

#if VC_IMPL_Scalar
#define VC_IMPL ::Vc::ScalarImpl
#elif VC_IMPL_AVX
#define VC_IMPL ::Vc::AVXImpl
#elif VC_IMPL_SSE4a
#define VC_IMPL ::Vc::SSE4aImpl
#elif VC_IMPL_SSE42
#define VC_IMPL ::Vc::SSE42Impl
#elif VC_IMPL_SSE41
#define VC_IMPL ::Vc::SSE41Impl
#elif VC_IMPL_SSSE3
#define VC_IMPL ::Vc::SSSE3Impl
#elif VC_IMPL_SSE3
#define VC_IMPL ::Vc::SSE3Impl
#elif VC_IMPL_SSE2
#define VC_IMPL ::Vc::SSE2Impl
#elif VC_IMPL_LRBni
#define VC_IMPL ::Vc::LRBniImpl
#endif

namespace Vc {
enum AlignedFlag {
    Aligned = 0
};
enum UnalignedFlag {
    Unaligned = 1
};
enum StreamingAndAlignedFlag { // implies Aligned
    Streaming = 2
};
enum StreamingAndUnalignedFlag {
    StreamingAndUnaligned = 3
};

static inline StreamingAndUnalignedFlag operator|(UnalignedFlag, StreamingAndAlignedFlag) { return StreamingAndUnaligned; }
static inline StreamingAndUnalignedFlag operator|(StreamingAndAlignedFlag, UnalignedFlag) { return StreamingAndUnaligned; }
static inline StreamingAndUnalignedFlag operator&(UnalignedFlag, StreamingAndAlignedFlag) { return StreamingAndUnaligned; }
static inline StreamingAndUnalignedFlag operator&(StreamingAndAlignedFlag, UnalignedFlag) { return StreamingAndUnaligned; }

static inline StreamingAndAlignedFlag operator|(AlignedFlag, StreamingAndAlignedFlag) { return Streaming; }
static inline StreamingAndAlignedFlag operator|(StreamingAndAlignedFlag, AlignedFlag) { return Streaming; }
static inline StreamingAndAlignedFlag operator&(AlignedFlag, StreamingAndAlignedFlag) { return Streaming; }
static inline StreamingAndAlignedFlag operator&(StreamingAndAlignedFlag, AlignedFlag) { return Streaming; }

enum Implementation {
    ScalarImpl, SSE2Impl, SSE3Impl, SSSE3Impl, SSE41Impl, SSE42Impl, SSE4aImpl, AVXImpl, LRBniImpl, LRBniPrototypeImpl
};
namespace Internal {
    template<Implementation Impl> struct HelperImpl;
    typedef HelperImpl<VC_IMPL> Helper;

    template<typename A> struct FlagObject;
    template<> struct FlagObject<AlignedFlag> { static inline AlignedFlag the() { return Aligned; } };
    template<> struct FlagObject<UnalignedFlag> { static inline UnalignedFlag the() { return Unaligned; } };
    template<> struct FlagObject<StreamingAndAlignedFlag> { static inline StreamingAndAlignedFlag the() { return Streaming; } };
    template<> struct FlagObject<StreamingAndUnalignedFlag> { static inline StreamingAndUnalignedFlag the() { return StreamingAndUnaligned; } };
} // namespace Internal
} // namespace Vc

#include "version.h"

#endif // VC_GLOBAL_H
