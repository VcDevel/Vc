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
 * The Vc library is a collection of vector classes with existing implementations for SSE, LRBni or
 * a scalar fallback.
 *
 * \li Vc::float_v
 * \li Vc::sfloat_v
 * \li Vc::double_v
 * \li Vc::int_v
 * \li Vc::uint_v
 * \li Vc::short_v
 * \li Vc::ushort_v
 *
 * \li Vc::float_m
 * \li Vc::sfloat_m
 * \li Vc::double_m
 * \li Vc::int_m
 * \li Vc::uint_m
 * \li Vc::short_m
 * \li Vc::ushort_m
 *
 * Per default, code compiled against the Vc headers will use the instruction set that the compiler
 * says is available. For example compiling with "g++ -mssse3" will enable compilation against the
 * SSE implementation using SSE the instruction sets SSE, SSE2, SSE3 and SSSE3. If you want to force
 * compilation against a specific implementation of the vector classes you can set the macro
 * VC_IMPL to either "Scalar", "SSE2", "SSE3", "SSSE3", "SSE4_1" or "LRBni". Setting VC_IMPL to
 * "SSE" will force the SSE instruction set but letting the headers figure out the version to use or
 * if that fails use SSE4.1.
 * After you include a Vc header you will have the following macros available that you can (but
 * normally should not) use to determine the implementation Vc uses:
 * \li VC_IMPL_Scalar
 * \li VC_IMPL_LRBni
 * \li VC_IMPL_SSE (shorthand for SSE2 || SSE3 || SSSE3 || SSE4_1. SSE1 alone is not supported.)
 * \li VC_IMPL_SSE2
 * \li VC_IMPL_SSE3
 * \li VC_IMPL_SSSE3
 * \li VC_IMPL_SSE4_1
 *
 * \todo
 *  write/link example code, document mask classes, document remaining vector functions
 */

/**
 * \defgroup Vectors
 *
 * Vector classes are abstractions for SIMD instructions.
 *
 * \defgroup Masks
 *
 * Mask classes are abstractions for the results of vector comparisons. The actual implementation
 * differs depending on the SIMD instruction set. On SSE they contain a full 128bit datatype while
 * on LRBni they are stored as 16bit unsigned integers.
 *
 * \defgroup Utilities
 *
 * Utilities that either extend the language or provide other useful functionality outside of the
 * classes.
 *
 * \defgroup Math
 *
 * Functions that implement math functions. Take care that some of the implementations will return
 * results with less precision than what the FPU calculates.
 */

/**
 * \brief Vector Classes
 *
 * Depending on preprocessing macros, the vector classes inside the Vc namespace will be implemented
 * with either
 * \li SSE vectors
 * \li LRBni vectors
 * \li scalar fallback vectors (i.e. scalars, but with the same API)
 */
namespace Vc
{
    /**
     * Enum to declare platform specific constants
     */
    enum {
        /**
         * Specifies the byte boundary for memory alignments necessary for aligned loads and stores.
         */
        VectorAlignment,

        /**
         * Special initializer for vector constructors to create a fast initialization to zero.
         */
        Zero,

        /**
         * Special initializer for vector constructors to create a fast initialization to 1.
         */
        One,

        /**
         * Special initializer for vector constructors to create a vector with the entries 0, 1, 2,
         * 3, 4, 5, ... (depending on the vectors size, of course).
         */
        IndexesFromZero
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
     * \note currently only has an effect for SSE vectors
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
