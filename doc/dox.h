/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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
 * The Vc library is a collection of vector classes with existing implementations for SSE, AVX,
 * and a scalar fallback.
 *
 * \ref intro
 *
 * \ref portability
 *
 * \ref featuremacros
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
 * VC_IMPL to either "Scalar", "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2", "SSE4a", or "AVX".
 * Setting VC_IMPL to
 * "SSE" will force the SSE instruction set, but lets the headers figure out the version to use or,
 * if that fails, uses SSE4.1.
 * After you include a Vc header, you will have the following macros available, which you can (but
 * normally should not) use to determine the implementation Vc uses:
 * \li \c VC_IMPL_Scalar
 * \li \c VC_IMPL_SSE (shorthand for SSE2 || SSE3 || SSSE3 || SSE4_1. SSE1 alone is not supported.)
 * \li \c VC_IMPL_SSE2
 * \li \c VC_IMPL_SSE3
 * \li \c VC_IMPL_SSSE3
 * \li \c VC_IMPL_SSE4_1
 * \li \c VC_IMPL_SSE4_2
 * \li \c VC_IMPL_SSE4a
 * \li \c VC_IMPL_AVX
 *
 * Another set of macros you may use for target specific implementations are the \c VC_*_V_SIZE
 * macros: \ref Utilities
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
 *
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
 * For completeness, note that you can optimize the division in the normalize function further:
 * \code
 *   const float_v d_inv = float_v::One() / Vc::sqrt(x * x + y * y + z * z);
 *   const float_v d_inv = Vc::rsqrt(x * x + y * y + z * z); // less accurate, but faster
 * \endcode
 * Then you can multiply \c x, \c y, and \c z with \c d_inv, which is considerably faster than three
 * divisions.
 *
 * As you can probably see, the new challenge with Vc is the use of good data-structures which
 * support horizontal vectorization. Depending on your problem at hand this may become the main
 * focus of design (it does not have to be, though).
 */

/**
 * \page portability Portability Issues
 *
 * One of the major goals of Vc is to ease development of portable code, while achieving highest
 * possible performance that requires target architecture specific instructions. This is possible
 * through having just a single type use different implementations of the same API depending on the
 * target architecture. Many of the details of the target architecture are often dependent on the
 * compiler flags that were used. Also there can be subtle differences between the implementations
 * that could lead to problems. This page aims to document all issues you might need to know about.
 *
 * \par Compiler Flags
 *
 * \li \e GCC: The compiler should be called with the -march=\<target\> flag. Take a look at the GCC
 * manpage to find all possibilities for \<target\>. Additionally it is best to also add the -msse2
 * -msse3 ... -mavx flags. If no SIMD instructions are enabled via compiler flags, Vc must fall back
 * to the scalar implementation.
 * \li \e Clang: The same as for GCC applies.
 * \li \e ICC: Same as GCC, but the flags are called -xAVX -xSSE4.2 -xSSE4.1 -xSSSE3 -xSSE3 -xSSE2.
 * \li \e MSVC: On 32bit you can add the /arch:SSE2 flag. That's about all the MSVC documentation
 * says. Still the MSVC compiler knows about the newer instructions in SSE3 and upwards. How you can
 * determine what CPUs will be supported by the resulting binary is unclear.
 *
 * \par Where does the final executable run?
 *
 * You must be aware of the fact that a binary that is built for a given SIMD hardware may not run
 * on a processor that does not have these instructions. The executable will work fine as long as no
 * such instruction is actually executed and only crash at the place where such an instruction is
 * used. Thus it is better to check at application start whether the compiled in SIMD hardware is
 * really supported on the executing CPU. This can be determined with the
 * currentImplementationSupported function.
 *
 * If you want to distribute a binary that runs correctly on many different systems you either must
 * restrict it to the least common denominator (which often is SSE2), or you must compile the code
 * several times, with the different target architecture compiler options. A simple way to combine
 * the resulting executables would be via a wrapping script/executable that determines the correct
 * executable to use. A more sophisticated option is the use of the ifunc attribute GCC provides.
 * Other compilers might provide similar functionality.
 *
 * \par Guarantees
 *
 * It is guaranteed that:
 * \li \code int_v::Size == uint_v::Size == float_v::Size \endcode
 * \li \code short_v::Size == ushort_v::Size == sfloat_v::Size \endcode
 *
 * \par Important Differences between Implementations
 *
 * \li Obviously the number of entries in a vector depends on the target architecture.
 * \li Because of the guarantees above, sfloat_v does not necessarily map to a single SIMD register
 * and thus there could be a higher register pressure when this type is used.
 * \li Hardware that does not support 16-Bit integer vectors can implement the short_v and ushort_v
 * API via 32-Bit integer vectors. Thus, some of the overflow behavior might be slightly different,
 * and truncation will only happen when the vector is stored to memory.
 */

/**
 * \page featuremacros Feature Macros
 *
 * The following macros are available to enable/disable selected features:
 * \li \e VC_NO_STD_FUNCTIONS
 * \li \e VC_CLEAN_NAMESPACE
 * \li \e VC_NO_AUTOMATIC_BOOL_FROM_MASK
 * \li \e VC_NO_VERSION_CHECK
 */

/**
 * \defgroup Vectors Vectors
 *
 * The vector classes abstract the SIMD registers and their according instructions into types that
 * feel very familiar to C++ developers.
 *
 * Note that the documented types Vc::float_v, Vc::double_v, Vc::int_v, Vc::uint_v, Vc::short_v,
 * and Vc::ushort_v are actually \c typedefs of the \c Vc::Vector<T> class:
 * \code
 * namespace Vc {
 *   template<typename T> class Vector;
 *   typedef Vector<double> double_v;
 *   typedef Vector<float> float_v;
 *   // ...
 * }
 * \endcode
 *
 * \par Some general information on using the vector classes:
 *
 * Generally you can always mix scalar values with vectors as Vc will automatically broadcast the
 * scalar to a vector and then execute a vector operation. But, in order to ensure that implicit
 * type conversions only happen as defined by the C standard, there is only a very strict implicit
 * scalar to vector constructor:
 * \code
 * int_v a = 1;     // good:             int_v(int)
 * uint_v b = 1u;   // good:             uint_v(unsigned int)
 * uint_v c = 1;    // does not compile: uint_v(int)
 * float_v d = 1;   // does not compile: float_v(int)
 * float_v e = 1.;  // does not compile: float_v(double)
 * float_v f = 1.f; // good:             float_v(float)
 * \endcode
 *
 * The following ways of initializing a vector are not allowed:
 * \code
 * int_v v(3, 2, 8, 0); // constructor does not exist because it is not portable
 * int_v v;
 * v[0] = 3; v[1] = 2; v[2] = 8; v[3] = 0; // do not hardcode the number of entries!
 * // You can not know whether somebody will compile with Vc Scalar where int_v::Size == 1
 * \endcode
 *
 * Instead, if really necessary you can do:
 * \code
 * Vc::int_v v;
 * for (int i = 0; i < int_v::Size; ++i) {
 *   v[i] = f(i);
 * }
 * // which is equivalent to:
 * v = int_v::IndexesFromZero().apply(f);
 * \endcode
 */

/**
 * \defgroup Masks Masks
 *
 * Mask classes are abstractions for the results of vector comparisons. The actual implementation
 * differs depending on the SIMD instruction set. On SSE they contain a full 128-bit datatype while
 * on a different architecture they might be bit-fields.
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
 * \ingroup Vectors
 * \brief Vector Classes Namespace
 *
 * All functions and types of Vc are defined inside the Vc namespace.
 *
 * To be precise, most types are actually defined inside a second namespace, such as Vc::SSE. At
 * compile-time the correct implementation is simply imported into the Vc namespace.
 */
namespace Vc
{
    /**
     * \class Vector dox.h <Vc/vector.h>
     * \ingroup Vectors
     *
     * The main vector class.
     *
     * \li Vc::float_v
     * \li Vc::sfloat_v
     * \li Vc::double_v
     * \li Vc::int_v
     * \li Vc::uint_v
     * \li Vc::short_v
     * \li Vc::ushort_v
     *
     * are only specializations of this class. For the full documentation take a look at the
     * specialized classes. For most cases there are no API differences for the specializations.
     * Thus you can make use of \c Vector<T> for generic programming.
     */
    template<typename T> class Vector
    {
        public:
#define INDEX_TYPE uint_v
#define VECTOR_TYPE Vector<T>
#define ENTRY_TYPE T
#define MASK_TYPE float_m
#define EXPONENT_TYPE int_v
#include "dox-common-ops.h"
#include "dox-real-ops.h"
#undef INDEX_TYPE
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef EXPONENT_TYPE
    };

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

#define INDEX_TYPE uint_v
#define VECTOR_TYPE float_v
#define ENTRY_TYPE float
#define MASK_TYPE float_m
#define EXPONENT_TYPE int_v
    /**
     * \class float_v dox.h <Vc/float_v>
     * \ingroup Vectors
     *
     * SIMD Vector of single precision floats.
     *
     * \note This is the same type as Vc::Vector<float>.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
#include "dox-real-ops.h"
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
     *
     * \note This is the same type as Vc::Vector<double>.
     */
    class VECTOR_TYPE
    {
        public:
#include "dox-common-ops.h"
#include "dox-real-ops.h"
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
     *
     * \note This is the same type as Vc::Vector<int>.
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
     *
     * \note This is the same type as Vc::Vector<unsigned int>.
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
     * \note This is the same type as Vc::Vector<short>.
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
     * \note This is the same type as Vc::Vector<unsigned short>.
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
#undef EXPONENT_TYPE

#define EXPONENT_TYPE short_v
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
#include "dox-real-ops.h"
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
#undef EXPONENT_TYPE
#undef VECTOR_TYPE
#undef ENTRY_TYPE
#undef MASK_TYPE
#undef INDEX_TYPE

    /**
     * \ingroup Math
     * \note Often int_v::Size == double_v::Size * 2, then only every second value in \p *e is defined.
     */
    double_v frexp(const double_v &x, int_v *e);
    /**
     * \ingroup Math
     * \note Often int_v::Size == double_v::Size * 2, then only every second value in \p *e is defined.
     */
    double_v ldexp(double_v x, int_v e);

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
 * Vc_foreach_bit(int i, a < 0.f) {
 *   std::cout << a[i] << "\n";
 * }
 * \endcode
 * The example prints all the values in \p a that are negative, and only those.
 *
 * \param iterator  The iterator variable. For example "int i".
 * \param mask      The mask to iterate over. You can also just write a vector operation that returns a
 *                  mask.
 */
#define Vc_foreach_bit(iterator, mask)

/**
 * \ingroup Utilities
 *
 * Alias for Vc_foreach_bit unless VC_CLEAN_NAMESPACE is defined.
 */
#define foreach_bit(iterator, mask)

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/IO>
 *
 * Prints the contents of a vector into a stream object.
 *
 * \code
 * const Vc::int_v v(Vc::IndexesFromZero);
 * std::cout << v << std::endl;
 * \endcode
 * will output (with SSE):
\verbatim
[0, 1, 2, 3]
\endverbatim
 *
 * \param s Any standard C++ ostream object. For example std::cout or a std::stringstream object.
 * \param v Any Vc::Vector object.
 * \return  The ostream object: to chain multiple stream operations.
 *
 * \note With the GNU standard library this function will check, whether the output stream is a tty.
 * In that case it will colorize the output.
 */
template<typename T>
std::ostream &operator<<(std::ostream &s, const Vc::Vector<T> &v);

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/IO>
 *
 * Prints the contents of a mask into a stream object.
 *
 * \code
 * const Vc::short_m m = Vc::short_v::IndexesFromZero() < 3;
 * std::cout << m << std::endl;
 * \endcode
 * will output (with SSE):
\verbatim
m[1110 0000]
\endverbatim
 *
 * \param s Any standard C++ ostream object. For example std::cout or a std::stringstream object.
 * \param v Any Vc mask object.
 * \return  The ostream object: to chain multiple stream operations.
 *
 * \note With the GNU standard library this function will check, whether the output stream is a tty.
 * In that case it will colorize the output.
 */
template<typename T>
std::ostream &operator<<(std::ostream &s, const typename Vc::Vector<T>::Mask &v);

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/IO>
 *
 * Prints the contents of a Memory object into a stream object.
 *
 * \code
 * Vc::Memory<int_v, 10> m;
 * for (int i = 0; i < m.entriesCount(); ++i) {
 *   m[i] = i;
 * }
 * std::cout << m << std::endl;
 * \endcode
 * will output (with SSE):
\verbatim
{[0, 1, 2, 3] [4, 5, 6, 7] [8, 9, 0, 0]}
\endverbatim
 *
 * \param s Any standard C++ ostream object. For example std::cout or a std::stringstream object.
 * \param m Any Vc::Memory object.
 * \return  The ostream object: to chain multiple stream operations.
 *
 * \note With the GNU standard library this function will check, whether the output stream is a tty.
 * In that case it will colorize the output.
 *
 * \warning Please do not forget that printing a large memory object can take a long time.
 */
template<typename V, typename Parent, typename Dimension, typename RM>
inline std::ostream &operator<<(std::ostream &s, const Vc::MemoryBase<V, Parent, Dimension, RM> &m);

namespace Vc
{
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/version.h>
 *
 * \returns the version string of the Vc headers.
 *
 * \note There exists a built-in check that ensures on application startup that the Vc version of the
 * library (link time) and the headers (compile time) are equal. A mismatch between headers and
 * library could lead to errors that are very hard to debug.
 * \note If you need to disable the check (it costs a very small amount of application startup time)
 * you can define VC_NO_VERSION_CHECK at compile time.
 */
const char *versionString();

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/version.h>
 *
 * \returns the version of the Vc headers encoded in an integer.
 */
unsigned int versionNumber();

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/version.h>
 *
 * Contains the version string of the Vc headers. Same as Vc::versionString().
 */
#define VC_VERSION_STRING

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/version.h>
 *
 * Contains the encoded version number of the Vc headers. Same as Vc::versionNumber().
 */
#define VC_VERSION_NUMBER

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/version.h>
 *
 * Helper macro to compare against an encoded version number.
 * Example:
 * \code
 * #if VC_VERSION_CHECK(0.5.1) >= VC_VERSION_NUMBER
 * \endcode
 */
#define VC_VERSION_CHECK(major, minor, patch)

/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a double_v.
 */
#define VC_DOUBLE_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a float_v.
 */
#define VC_FLOAT_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a sfloat_v.
 */
#define VC_SFLOAT_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a int_v.
 */
#define VC_INT_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a uint_v.
 */
#define VC_UINT_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a short_v.
 */
#define VC_SHORT_V_SIZE
/**
 * \ingroup Utilities
 * \headerfile dox.h <Vc/vector.h>
 * An integer (for use with the preprocessor) that gives the number of entries in a ushort_v.
 */
#define VC_USHORT_V_SIZE

} // namespace Vc
