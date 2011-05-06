/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

/**
 * \mainpage
 * \image html logo.png
 *
 * The Vc library is a collection of vector classes with existing implementations for SSE, AVX, LRBni,
 * and a scalar fallback.
 *
 * \subpage intro
 *
 * \li \ref Vectors
 * \li \ref Masks
 * \li \ref Utilities
 * \li \ref Math
 *
 * Per default, code compiled against the Vc headers will use the instruction set that the compiler
 * says is available. For example compiling with "g++ -mssse3" will enable compilation against the
 * SSE implementation using SSE the instruction sets SSE, SSE2, SSE3 and SSSE3. If you want to force
 * compilation against a specific implementation of the vector classes you can set the macro
 * VC_IMPL to either "Scalar", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "SSE4a", "AVX" or "LRBni".
 * Setting VC_IMPL to
 * "SSE" will force the SSE instruction set, but lets the headers figure out the version to use or,
 * if that fails, uses SSE4.1.
 * After you include a Vc header, you will have the following macros available, which you can (but
 * normally should not) use to determine the implementation Vc uses:
 * \li VC_IMPL_Scalar
 * \li VC_IMPL_LRBni
 * \li VC_IMPL_SSE (shorthand for SSE2 || SSE3 || SSSE3 || SSE4_1. SSE1 alone is not supported.)
 * \li VC_IMPL_SSE2
 * \li VC_IMPL_SSE3
 * \li VC_IMPL_SSSE3
 * \li VC_IMPL_SSE4_1
 * \li VC_IMPL_SSE4_2
 * \li VC_IMPL_SSE4a
 * \li VC_IMPL_AVX
 */

/**
 * \page intro Introduction
 *
 * If you are new to vectorization please read this following part and make sure you understand it:
 * \li Forget what you learned about vectors in math classes. SIMD vectors are a different concept!
 * \li Forget about containers that also go by the name of a vector. SIMD vectors are a different concept!
 * \li A vector is defined by the hardware as a special register which is wider than required for a
 * single value. Thus multiple values fit into one register. The width of this register and the
 * size of the scalar data type used normally determine the number of entries in the vector, and
 * thus this number is an unchangeable property of the hardware and therefore not a variable in the
 * Vc API.
 * \li Note that hardware is free to use different vector register widths for different data types.
 * For example AVX has instructions for 256-bit floating point registers, but only 128-bit integer
 * instructions.
 *
 * \par Example 1:
 * You can modify a function to use vector types and thus implement a horizontal vectorization. The
 * original scalar function could look like this:
 * \code
 * void normalize(float &x, float &y, float &z)
 * {
 *   const float d = std::sqrt(x * x + y * y + z * z);
 *   x /= d;
 *   y /= d;
 *   z /= d;
 * }
 * \endcode
 * To vectorize it with Vc the types must be substituted by their Vc counterparts and math functions
 * must simply use the Vc implementation which is not part of the \c std namespace:
 * \code
 * using Vc::float_v;
 *
 * void normalize(float_v &x, float_v &y, float_v &z)
 * {
 *   const float_v d = Vc::sqrt(x * x + y * y + z * z);
 *   x /= d;
 *   y /= d;
 *   z /= d;
 * }
 * \endcode
 * The latter function is able to normalize four 3D vectors when compiled for SSE in the same
 * time the former function normalizes one 3D vector.
 *
 * \par
 * As you can probably see, the new challenge with Vc is the use of good data-structures which
 * support horizontal vectorization. Depending on your problem at hand this may become the main
 * focus of design (it does not have to be, though).
 */

/**
 * \defgroup Vectors Vectors
 *
 * Vector classes are abstractions for SIMD instructions.
 */

/**
 * \defgroup Masks Masks
 *
 * Mask classes are abstractions for the results of vector comparisons. The actual implementation
 * differs depending on the SIMD instruction set. On SSE they contain a full 128-bit datatype while
 * on LRBni they are stored as 16-bit unsigned integers.
 */

/**
 * \defgroup Utilities Utilities
 *
 * Utilities that either extend the language or provide other useful functionality outside of the
 * classes.
 */

/**
 * \defgroup Math Math
 *
 * Functions that implement math functions. Take care that some of the implementations will return
 * results with less precision than what the FPU calculates.
 */

/**
 * \brief Vector Classes Namespace
 *
 * All functions and types of Vc are defined inside the Vc namespace.
 */
namespace Vc
{
    /**
     * \ingroup Vectors
     *
     * Enum to declare platform specific constants
     */
    enum PlatformConstants {
        /**
         * Specifies the byte boundary for memory alignments necessary for aligned loads and stores.
         */
        VectorAlignment
    };

    /**
     * \ingroup Vectors
     *
     * Enum to declare special initializers for vector constructors.
     */
    enum SpecialInitializer {
        /**
         * Used for optimized construction of vectors initialized to zero.
         */
        Zero,

        /**
         * Used for optimized construction of vectors initialized to one.
         */
        One,

        /**
         * Parameter to create a vector with the entries 0, 1, 2,
         * 3, 4, 5, ... (depending on the vector's size, of course).
         */
        IndexesFromZero
    };

    /**
     * \ingroup Vectors
     *
     * Enum for load and store functions to select the optimizations that are safe to use.
     */
    enum LoadStoreFlags {
        /**
         * Tells Vc that the load/store can expect a memory address that is aligned on the correct
         * boundary.
         *
         * If you specify Aligned, but the memory address is not aligned the program will most
         * likely crash.
         */
        Aligned,

        /**
         * Tells Vc that the load/store can \em not expect a memory address that is aligned on the correct
         * boundary.
         *
         * If you specify Unaligned, but the memory address is aligned the load/store will execute
         * slightly slower than necessary.
         */
        Unaligned,

        /**
         * Tells Vc to bypass the cache for the load/store. Whether this will actually be done
         * depends on the instruction set in use.
         *
         * Streaming stores can be interesting when the code calculates values that, after being
         * written to memory, will not be used for a long time or used by a different thread.
         *
         * \note Passing Streaming as only alignment flag implies Aligned! If you need unaligned
         * memory access you can use
         * \code
         * v.store(mem, Vc::Unaligned | Vc::Streaming);
         * \endcode
         */
        Streaming
    };

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

#define INDEX_TYPE uint_v
#define VECTOR_TYPE float_v
#define ENTRY_TYPE float
#define MASK_TYPE float_m
    /**
     * \class float_v dox.h <Vc/float_v>
     * \ingroup Vectors
     *
     * SIMD Vector of single precision floats.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class float_m dox.h <Vc/float_v>
     * \ingroup Masks
     *
     * Mask object to use with float_v objects.
     *
     * Of the same type as int_m and uint_m.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#include "dox-math.h"
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE

#define VECTOR_TYPE double_v
#define ENTRY_TYPE double
#define MASK_TYPE double_m
    /**
     * \class double_v dox.h <Vc/double_v>
     * \ingroup Vectors
     *
     * SIMD Vector of double precision floats.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class double_m dox.h <Vc/double_v>
     * \ingroup Masks
     *
     * Mask object to use with double_v objects.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#include "dox-math.h"
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE

#define VECTOR_TYPE int_v
#define ENTRY_TYPE int
#define MASK_TYPE int_m
#define INTEGER
    /**
     * \class int_v dox.h <Vc/int_v>
     * \ingroup Vectors
     *
     * SIMD Vector of 32 bit signed integers.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class int_m dox.h <Vc/int_v>
     * \ingroup Masks
     *
     * Mask object to use with int_v objects.
     *
     * Of the same type as float_m and uint_m.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE

#define VECTOR_TYPE uint_v
#define ENTRY_TYPE unsigned int
#define MASK_TYPE uint_m
    /**
     * \class uint_v dox.h <Vc/uint_v>
     * \ingroup Vectors
     *
     * SIMD Vector of 32 bit unsigned integers.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class uint_m dox.h <Vc/uint_v>
     * \ingroup Masks
     *
     * Mask object to use with uint_v objects.
     *
     * Of the same type as int_m and float_m.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef INDEX_TYPE

#define INDEX_TYPE ushort_v
#define VECTOR_TYPE short_v
#define ENTRY_TYPE short
#define MASK_TYPE short_m
    /**
     * \class short_v dox.h <Vc/short_v>
     * \ingroup Vectors
     *
     * SIMD Vector of 16 bit signed integers.
     *
     * \warning Vectors of this type are not supported on all platforms. In that case the vector
     * class will silently fall back to a Vc::int_v.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class short_m dox.h <Vc/short_v>
     * \ingroup Masks
     *
     * Mask object to use with short_v objects.
     *
     * Of the same type as ushort_m.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE

#define VECTOR_TYPE ushort_v
#define ENTRY_TYPE unsigned short
#define MASK_TYPE ushort_m
    /**
     * \class ushort_v dox.h <Vc/ushort_v>
     * \ingroup Vectors
     *
     * SIMD Vector of 16 bit unsigned integers.
     *
     * \warning Vectors of this type are not supported on all platforms. In that case the vector
     * class will silently fall back to a Vc::uint_v.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class ushort_m dox.h <Vc/ushort_v>
     * \ingroup Masks
     *
     * Mask object to use with ushort_v objects.
     *
     * Of the same type as short_m.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef INTEGER

#define VECTOR_TYPE sfloat_v
#define ENTRY_TYPE float
#define MASK_TYPE sfloat_m
    /**
     * \class sfloat_v dox.h <Vc/sfloat_v>
     * \ingroup Vectors
     *
     * SIMD Vector of single precision floats that is guaranteed to have as many entries as a
     * Vc::short_v / Vc::ushort_v.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
    };
    /**
     * \class sfloat_m dox.h <Vc/sfloat_v>
     * \ingroup Masks
     * \ingroup Masks
     *
     * Mask object to use with sfloat_v objects.
     */
    class MASK_TYPE
    {
        public:
#include "dox-common-mask-ops.h"
    };
#include "dox-math.h"
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef INDEX_TYPE

    /**
     * \ingroup Utilities
     *
     * Force the vectors passed to the function into registers. This can be useful after looking at
     * the emitted assembly to force the compiler to optimize properly.
     *
     * \note Currently only has an effect for SSE vectors.
     * \note MSVC does not support this function at all.
     *
     * \warning Be careful with this function, especially since it can render the compiler unable to
     * compile for 32 bit systems if it forces more than 8 vectors in registers.
     */
    void forceToRegisters(const vec &, ...);
}

/**
 * \ingroup Utilities
 *
 * Loop over all set bits in the mask. The iterator variable will be set to the position of the set
 * bits. A mask of e.g. 00011010 would result in the loop being called with the iterator being set to
 * 1, 3, and 4.
 *
 * This allows you to write:
 * \code
 * float_v a = ...;
 * foreach_bit(int i, a < 0.f) {
 *   std::cout << a[i] << "\n";
 * }
 * \endcode
 * The example prints all the values in \p a that are negative, and only those.
 *
 * \param iterator  The iterator variable. For example "int i".
 * \param mask      The mask to iterate over. You can also just write a vector operation that returns a
 *                  mask.
 */
#define foreach_bit(iterator, mask)
