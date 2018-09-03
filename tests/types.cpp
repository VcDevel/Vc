/*{{{
    Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#include "unittest.h"

#if defined Vc_GCC && Vc_GCC >= 0x60000
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#define ALL_TYPES concat<AllVectors, SimdArrays<32>>

TEST_TYPES(V, check_EntryType, ALL_TYPES)
{
    V v = V();
    auto scalar = v.sum();
    static_assert(std::is_same<typename V::EntryType, decltype(scalar)>::value, "");
    static_assert(std::is_same<typename V::value_type, decltype(scalar)>::value, "");
}

TEST_TYPES(V, check_VectorType, AllVectors)
{
    V v = V();
    auto internalData = v.data();
    static_assert(std::is_same<typename V::VectorType, decltype(internalData)>::value, "");
    static_assert(std::is_same<typename V::vector_type, decltype(internalData)>::value, "");
}

TEST_TYPES(V, check_MaskType, ALL_TYPES)
{
    V v = V();
    auto mask = v == v;
    static_assert(std::is_same<typename V::MaskType, decltype(mask)>::value, "");
    static_assert(std::is_same<typename V::mask_type, decltype(mask)>::value, "");
}

TEST_TYPES(V, check_IndexType, ALL_TYPES)
{
    using IndexType = typename V::IndexType;
    static_assert(Vc::Traits::isSimdArray<IndexType>::value,
                  "IndexType is not a SimdArray instance");
    static_assert(IndexType::Size >= V::Size,
                  "IndexType does not have the expected minimum size");
    static_assert(std::is_same<typename IndexType::value_type, int>::value,
                  "IndexType does not have the expected value_type");
}

static_assert(!Vc::is_simd_vector<Vc::int64_v>::value, "");

TEST_TYPES(V, checkIntegerType, Vc::int64_v, Vc::uint64_v, Vc::int32_v, Vc::uint32_v,
           Vc::int16_v, Vc::uint16_v, Vc::int8_v, Vc::uint8_v)
{
    V v;
    MEMCOMPARE(v, v);
}
