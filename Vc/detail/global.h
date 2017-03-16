/*  This file is part of the Vc library. {{{
Copyright © 2009-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_GLOBAL_H_
#define VC_GLOBAL_H_

#include "../fwddecl.h"
#include "macros.h"
#include <cstdint>

Vc_VERSIONED_NAMESPACE_BEGIN

typedef   signed char        int8_t;
typedef unsigned char       uint8_t;
typedef   signed short      int16_t;
typedef unsigned short     uint16_t;
typedef   signed int        int32_t;
typedef unsigned int       uint32_t;
typedef   signed long long  int64_t;
typedef unsigned long long uint64_t;

/**
 * \ingroup Utilities
 *
 * Enum that specifies the alignment and padding restrictions to use for memory allocation with
 * Vc::malloc.
 */
enum MallocAlignment {
    /**
     * Align on boundary of vector sizes (e.g. 16 Bytes on SSE platforms) and pad to allow
     * vector access to the end. Thus the allocated memory contains a multiple of
     * VectorAlignment bytes.
     */
    AlignOnVector,
    /**
     * Align on boundary of cache line sizes (e.g. 64 Bytes on x86) and pad to allow
     * full cache line access to the end. Thus the allocated memory contains a multiple of
     * 64 bytes.
     */
    AlignOnCacheline,
    /**
     * Align on boundary of page sizes (e.g. 4096 Bytes on x86) and pad to allow
     * full page access to the end. Thus the allocated memory contains a multiple of
     * 4096 bytes.
     */
    AlignOnPage
};

/**
 * \ingroup Utilities
 *
 * Enum to identify a certain SIMD instruction set.
 *
 * You can use \ref CurrentImplementation for the currently active implementation.
 *
 * \see ExtraInstructions
 */
enum Implementation : std::uint_least32_t { // TODO: make enum class
    /// uses only fundamental types
    ScalarImpl,
    /// x86 SSE + SSE2
    SSE2Impl,
    /// x86 SSE + SSE2 + SSE3
    SSE3Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3
    SSSE3Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3 + SSE4.1
    SSE41Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3 + SSE4.1 + SSE4.2
    SSE42Impl,
    /// x86 AVX
    AVXImpl,
    /// x86 AVX + AVX2
    AVX2Impl,
    /// Intel Xeon Phi
    MICImpl,
    /// ARM NEON
    NeonImpl,
    ImplementationMask = 0xfff };

/**
 * \ingroup Utilities
 *
 * The list of available instructions is not easily described by a linear list of instruction sets.
 * On x86 the following instruction sets always include their predecessors:
 * SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2
 *
 * But there are additional instructions that are not necessarily required by this list. These are
 * covered in this enum.
 */
enum ExtraInstructions : std::uint_least32_t { // TODO: make enum class
    //! Support for float16 conversions in hardware
    Float16cInstructions  = 0x01000,
    //! Support for FMA4 instructions
    Fma4Instructions      = 0x02000,
    //! Support for XOP instructions
    XopInstructions       = 0x04000,
    //! Support for the population count instruction
    PopcntInstructions    = 0x08000,
    //! Support for SSE4a instructions
    Sse4aInstructions     = 0x10000,
    //! Support for FMA instructions (3 operand variant)
    FmaInstructions       = 0x20000,
    //! Support for ternary instruction coding (VEX)
    VexInstructions       = 0x40000,
    //! Support for BMI2 instructions
    Bmi2Instructions      = 0x80000,
    // PclmulqdqInstructions,
    // AesInstructions,
    // RdrandInstructions
    ExtraInstructionsMask = 0xfffff000u
};

/**
 * \ingroup Utilities
 * This class identifies the specific implementation %Vc uses in the current translation
 * unit in terms of a type.
 *
 * Most importantantly, the type \ref CurrentImplementation instantiates the class
 * template with the bitmask identifying the current implementation. The contents of the
 * bitmask can be queried with the static member functions of the class.
 */
template <unsigned int Features> struct ImplementationT {
    /// Returns the currently used Vc::Implementation.
    static constexpr Implementation current()
    {
        return static_cast<Implementation>(Features & ImplementationMask);
    }
    /// Returns whether \p impl is the current Vc::Implementation.
    static constexpr bool is(Implementation impl)
    {
        return static_cast<unsigned int>(impl) == current();
    }
    /**
     * Returns whether the current Vc::Implementation implements at least \p low and at
     * most \p high.
     */
    static constexpr bool is_between(Implementation low, Implementation high)
    {
        return static_cast<unsigned int>(low) <= current() &&
               static_cast<unsigned int>(high) >= current();
    }
    /**
     * Returns whether the current code would run on a CPU providing \p extraInstructions.
     */
    static constexpr bool runs_on(unsigned int extraInstructions)
    {
        return (extraInstructions & Features & ExtraInstructionsMask) ==
               (Features & ExtraInstructionsMask);
    }
};
/**
 * \ingroup Utilities
 * Identifies the ISA extensions used in the current translation unit.
 *
 * \see ImplementationT
 */
using CurrentImplementation = ImplementationT<
#if defined Vc_HAVE_NEON
    NeonImpl
#elif defined(Vc_HAVE_MIC)
    MICImpl
#elif defined(Vc_HAVE_AVX2)
    AVX2Impl
#elif defined(Vc_HAVE_AVX)
    AVXImpl
#elif defined(Vc_HAVE_SSE4_2)
    SSE42Impl
#elif defined(Vc_HAVE_SSE4_1)
    SSE41Impl
#elif defined(Vc_HAVE_SSSE3)
    SSSE3Impl
#elif defined(Vc_HAVE_SSE3)
    SSE3Impl
#elif defined(Vc_HAVE_SSE2)
    SSE2Impl
#endif
#ifdef Vc_HAVE_SSE4a
    + Vc::Sse4aInstructions
#ifdef Vc_HAVE_XOP
    + Vc::XopInstructions
#ifdef Vc_HAVE_FMA4
    + Vc::Fma4Instructions
#endif
#endif
#endif
#ifdef Vc_HAVE_POPCNT
    + Vc::PopcntInstructions
#endif
#ifdef Vc_HAVE_FMA
    + Vc::FmaInstructions
#endif
#ifdef Vc_HAVE_BMI2
    + Vc::Bmi2Instructions
#endif
#ifdef Vc_USE_VEX_CODING
    + Vc::VexInstructions
#endif
    >;

/**
 * \ingroup Utilities
 * \headerfile version.h <Vc/version.h>
 *
 * \returns the version string of the %Vc headers.
 *
 * \note There exists a built-in check that ensures on application startup that the %Vc version of the
 * library (link time) and the headers (compile time) are equal. A mismatch between headers and
 * library could lead to errors that are very hard to debug.
 * \note If you need to disable the check (it costs a very small amount of application startup time)
 * you can define Vc_NO_VERSION_CHECK at compile time.
 */
inline const char *versionString() { return Vc_VERSION_STRING; }

/**
 * \ingroup Utilities
 * \headerfile version.h <Vc/version.h>
 *
 * \returns the version of the %Vc headers encoded in an integer.
 */
constexpr unsigned int versionNumber() { return Vc_VERSION_NUMBER; }

Vc_VERSIONED_NAMESPACE_END

#include "../version.h"

#endif // VC_GLOBAL_H_

// vim: foldmethod=marker
