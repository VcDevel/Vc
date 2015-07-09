/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#ifndef VC_COMMON_SIMDARRAYFWD_H_
#define VC_COMMON_SIMDARRAYFWD_H_

#include "utility.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{
/// \addtogroup simdarray
/// @{
/*select_best_vector_type{{{*/
namespace internal
{
/**
 * \internal
 * AVX::Vector<T> with T int, uint, short, or ushort is either two SSE::Vector<T> or the same as
 * SSE::Vector<T>. Thus we can skip AVX::Vector<T> for integral types altogether.
 */
template <typename T> struct never_best_vector_type : public std::false_type {};

// the AVX namespace only exists in AVX compilations, otherwise it's AVX2 - which is fine
#if defined(VC_IMPL_AVX) && !defined(VC_IMPL_AVX2)
template <typename T> struct never_best_vector_type<AVX::Vector<T>> : public std::is_integral<T> {};
#endif
}  // namespace internal

/**
 * \internal
 * Selects the best SIMD type out of a typelist to store N scalar values.
 */
template<std::size_t N, typename... Typelist> struct select_best_vector_type_impl;

template<std::size_t N, typename T> struct select_best_vector_type_impl<N, T>
{
    using type = T;
};
template<std::size_t N, typename T, typename... Typelist> struct select_best_vector_type_impl<N, T, Typelist...>
{
    using type = typename std::conditional<(N < T::Size || internal::never_best_vector_type<T>::value),
                                           typename select_best_vector_type_impl<N, Typelist...>::type,
                                           T>::type;
};
template <typename T, std::size_t N>
using select_best_vector_type =
    typename select_best_vector_type_impl<N,
#ifdef VC_IMPL_AVX2
                                          Vc::AVX2::Vector<T>,
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_AVX)
                                          Vc::AVX::Vector<T>,
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_SSE)
                                          Vc::SSE::Vector<T>,
                                          Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_MIC)
                                          Vc::MIC::Vector<T>,
                                          Vc::Scalar::Vector<T>
#endif
                                          >::type;
//}}}
/// @}
}  // namespace Common

// === having simdarray<T, N> in the Vc namespace leads to a ABI bug ===
//
// simdarray<double, 4> can be { double[4] }, { __m128d[2] }, or { __m256d } even though the type
// is the same.
// The question is, what should simdarray focus on?
// a) A type that makes interfacing between different implementations possible?
// b) Or a type that makes fixed size SIMD easier and efficient?
//
// a) can be achieved by using a union with T[N] as one member. But this may have more serious
// performance implications than only less efficient parameter passing (because compilers have a
// much harder time wrt. aliasing issues). Also alignment would need to be set to the sizeof in
// order to be compatible with targets with larger alignment requirements.
// But, the in-memory representation of masks is not portable. Thus, at the latest with AVX-512,
// there would be a problem with requiring simd_mask_array<T, N> to be an ABI compatible type.
// AVX-512 uses one bit per boolean, whereas SSE/AVX use sizeof(T) Bytes per boolean. Conversion
// between the two representations is not a trivial operation. Therefore choosing one or the other
// representation will have a considerable impact for the targets that do not use this
// representation. Since the future probably belongs to one bit per boolean representation, I would
// go with that choice.
//
// b) requires that simdarray<T, N> != simdarray<T, N> if
// simdarray<T, N>::vector_type != simdarray<T, N>::vector_type
//
// Therefore use simdarray<T, N, V>, where V follows from the above.
template <
    typename T, std::size_t N,
    typename VectorType = Common::select_best_vector_type<T, N>,
    std::size_t VectorSize = VectorType::size()  // this last parameter is only used for
                                                 // specialization of N == VectorSize
    >
class
#ifndef VC_ICC
    alignas((((Common::nextPowerOfTwo((N + VectorSize - 1) / VectorSize) *
               sizeof(VectorType)) -
              1) &
             127) +
            1)
#endif
        simdarray;

template <
    typename T, std::size_t N,
    typename VectorType = Common::select_best_vector_type<T, N>,
    std::size_t VectorSize = VectorType::size()  // this last parameter is only used for
                                                 // specialization of N == VectorSize
    >
class
#ifndef VC_ICC
    alignas((((Common::nextPowerOfTwo((N + VectorSize - 1) / VectorSize) *
               sizeof(typename VectorType::Mask)) -
              1) &
             127) +
            1)
#endif
        simd_mask_array;

/** \internal
 * Simple traits for simdarray to easily access internal types of non-atomic simdarray
 * types.
 */
template <typename T, std::size_t N> struct simdarray_traits {
    static constexpr std::size_t N0 = Common::left_size(N);
    static constexpr std::size_t N1 = Common::right_size(N);

    using storage_type0 = simdarray<T, N0>;
    using storage_type1 = simdarray<T, N1>;
};

template <typename T, std::size_t N, typename VectorType, std::size_t VectorSize>
Vc_INTRINSIC_L typename simdarray_traits<T, N>::storage_type0 &internal_data0(
    simdarray<T, N, VectorType, VectorSize> &x) Vc_INTRINSIC_R;
template <typename T, std::size_t N, typename VectorType, std::size_t VectorSize>
Vc_INTRINSIC_L typename simdarray_traits<T, N>::storage_type1 &internal_data1(
    simdarray<T, N, VectorType, VectorSize> &x) Vc_INTRINSIC_R;
template <typename T, std::size_t N, typename VectorType, std::size_t VectorSize>
Vc_INTRINSIC_L const typename simdarray_traits<T, N>::storage_type0 &internal_data0(
    const simdarray<T, N, VectorType, VectorSize> &x) Vc_INTRINSIC_R;
template <typename T, std::size_t N, typename VectorType, std::size_t VectorSize>
Vc_INTRINSIC_L const typename simdarray_traits<T, N>::storage_type1 &internal_data1(
    const simdarray<T, N, VectorType, VectorSize> &x) Vc_INTRINSIC_R;

template <typename T, std::size_t N, typename V>
Vc_INTRINSIC_L V &internal_data(simdarray<T, N, V, N> &x) Vc_INTRINSIC_R;
template <typename T, std::size_t N, typename V>
Vc_INTRINSIC_L const V &internal_data(const simdarray<T, N, V, N> &x) Vc_INTRINSIC_R;

namespace Traits
{
template <typename T, std::size_t N, typename V> struct is_atomic_simdarray_internal<simdarray<T, N, V, N>> : public std::true_type {};
template <typename T, std::size_t N, typename V> struct is_atomic_simd_mask_array_internal<simd_mask_array<T, N, V, N>> : public std::true_type {};

template <typename T, std::size_t N, typename VectorType, std::size_t M> struct is_simdarray_internal<simdarray<T, N, VectorType, M>> : public std::true_type {};
template <typename T, std::size_t N, typename VectorType, std::size_t M> struct is_simd_mask_array_internal<simd_mask_array<T, N, VectorType, M>> : public std::true_type {};
template <typename T, std::size_t N, typename V, std::size_t M> struct is_integral_internal      <simdarray<T, N, V, M>, false> : public std::is_integral<T> {};
template <typename T, std::size_t N, typename V, std::size_t M> struct is_floating_point_internal<simdarray<T, N, V, M>, false> : public std::is_floating_point<T> {};
template <typename T, std::size_t N, typename V, std::size_t M> struct is_signed_internal        <simdarray<T, N, V, M>, false> : public std::is_signed<T> {};
template <typename T, std::size_t N, typename V, std::size_t M> struct is_unsigned_internal      <simdarray<T, N, V, M>, false> : public std::is_unsigned<T> {};

template<typename T, std::size_t N> struct has_no_allocated_data_impl<Vc::simdarray<T, N>> : public std::true_type {};
}  // namespace Traits

}  // namespace Vc

#include "undomacros.h"

#endif  // VC_COMMON_SIMDARRAYFWD_H_

// vim: foldmethod=marker
