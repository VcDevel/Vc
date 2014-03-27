/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_COMMON_SIMD_MASK_ARRAY_H
#define VC_COMMON_SIMD_MASK_ARRAY_H

#include <type_traits>
#include <array>
#include "simd_array_data.h"
#include "utility.h"

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

template <typename T,
          std::size_t N,
          typename VectorType = typename Common::select_best_vector_type<N,
#ifdef VC_IMPL_AVX
                                                                         Vc::Vector<T>,
                                                                         Vc::SSE::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                                                         Vc::Vector<T>
#else
                                                                         Vc::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#endif
                                                                         >::type,
          std::size_t VectorSize = VectorType::size()  // this last parameter is only used for
                                                       // specialization of N == VectorSize
          >
class simd_mask_array;

template <typename T, std::size_t N, typename VectorType> class simd_mask_array<T, N, VectorType, N>
{
    using vector_type = VectorType;

public:
    using mask_type = typename vector_type::Mask;
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename mask_type::VectorEntryType;
    using value_type = typename mask_type::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;

    FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    simd_mask_array() = default;

    template <typename... Args>
    Vc_INTRINSIC explicit simd_mask_array(Args &&... args)
        : data(std::forward<Args>(args)...)
    {
    }

    /*
    template <typename U>
    Vc_INTRINSIC simd_mask_array(const simd_mask_array<U, N> &rhs)
        : data(sse_cast<__m128>(internal::mask_cast<simd_mask_array<U>::Size, Size>(rhs.dataI())))
    {
    }

    template <typename U, std::size_t M>
    Vc_INTRINSIC explicit simd_mask_array(
        const simd_mask_array<U, M> &rhs,
        typename std::enable_if<!is_implicit_cast_allowed_mask<U, T>::value, void *>::type =
            nullptr)
        : data(sse_cast<__m128>(internal::mask_cast<simd_mask_array<U>::Size, Size>(rhs.dataI())))
    {
    }
    */

    Vc_INTRINSIC void load(const bool *mem) { data.load(mem); }
    template <typename Flags> Vc_INTRINSIC void load(const bool *mem, Flags f)
    {
        data.load(mem, f);
    }

    Vc_INTRINSIC void store(bool *mem) const { data.store(mem); }
    template <typename Flags> Vc_INTRINSIC void store(bool *mem, Flags f) const
    {
        data.store(mem, f);
    }

    Vc_INTRINSIC Vc_PURE bool operator==(const simd_mask_array &rhs) const
    {
        return data == rhs.data;
    }
    Vc_INTRINSIC Vc_PURE bool operator!=(const simd_mask_array &rhs) const
    {
        return data != rhs.data;
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator!() const
    {
        return {!data};
    }

    Vc_INTRINSIC simd_mask_array &operator&=(const simd_mask_array &rhs)
    {
        data &= rhs.data;
        return *this;
    }
    Vc_INTRINSIC simd_mask_array &operator|=(const simd_mask_array &rhs)
    {
        data |= rhs.data;
        return *this;
    }
    Vc_INTRINSIC simd_mask_array &operator^=(const simd_mask_array &rhs)
    {
        data ^= rhs.data;
        return *this;
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator&(const simd_mask_array &rhs) const
    {
        return {data & rhs.data};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator|(const simd_mask_array &rhs) const
    {
        return {data | rhs.data};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator^(const simd_mask_array &rhs) const
    {
        return {data ^ rhs.data};
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator&&(const simd_mask_array &rhs) const
    {
        return {data && rhs.data};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator||(const simd_mask_array &rhs) const
    {
        return {data || rhs.data};
    }

    Vc_INTRINSIC Vc_PURE bool isFull() const { return data.isFull(); }
    Vc_INTRINSIC Vc_PURE bool isNotEmpty() const { return data.isNotEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isEmpty() const { return data.isEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isMix() const { return data.isMix(); }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
    Vc_INTRINSIC Vc_PURE operator bool() const { return isFull(); }
#endif

    Vc_INTRINSIC Vc_PURE int shiftMask() const { return data.shiftMask(); }

    Vc_INTRINSIC Vc_PURE int toInt() const { return data.toInt(); }

    // Vc_INTRINSIC decltype(std::declval<Storage &>().m(0)) operator[](size_t index) { return
    // data.m(index); }
    Vc_INTRINSIC Vc_PURE bool operator[](size_t index) const { return data[index]; }

    Vc_INTRINSIC Vc_PURE int count() const { return data.count(); }

    /**
     * Returns the index of the first one in the mask.
     *
     * The return value is undefined if the mask is empty.
     */
    Vc_INTRINSIC Vc_PURE int firstOne() const { return data.firstOne(); }

    /// \internal
    Vc_INTRINSIC simd_mask_array(mask_type &&x) : data(std::move(x)) {}

private:
    mask_type data;
};

template <typename T, std::size_t N, typename VectorType, std::size_t> class simd_mask_array
{
    static constexpr std::size_t N0 = Common::nextPowerOfTwo(N - N / 2);

    using storage_type0 = simd_mask_array<T, N0>;
    using storage_type1 = simd_mask_array<T, N - N0>;

    using Split = Common::Split<storage_type0::size()>;
    using vector_type = VectorType;

public:
    using mask_type = simd_mask_array;
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename storage_type0::VectorEntryType;
    using value_type = typename storage_type0::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;

    FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    simd_mask_array() = default;

    // default copy ctor/operator
    simd_mask_array(const simd_mask_array &) = default;
    simd_mask_array(simd_mask_array &&) = default;
    simd_mask_array &operator=(const simd_mask_array &) = default;
    simd_mask_array &operator=(simd_mask_array &&) = default;

    template <typename... Args>
    Vc_INTRINSIC explicit simd_mask_array(Args &&... args)
        : data0(Split::lo(std::forward<Args>(args))...)
        , data1(Split::hi(std::forward<Args>(args))...)
    {
    }

    template <typename Flags = DefaultLoadTag>
    Vc_INTRINSIC explicit simd_mask_array(const bool *mem, Flags f = Flags())
        : data0(mem, f), data1(mem + storage_type0::size(), f)
    {
    }

    Vc_INTRINSIC void load(const bool *mem)
    {
        data0.load(mem);
        data1.load(mem + storage_type0::size());
    }
    template <typename Flags> Vc_INTRINSIC void load(const bool *mem, Flags f)
    {
        data0.load(mem, f);
        data1.load(mem + storage_type0::size(), f);
    }

    Vc_INTRINSIC void store(bool *mem) const
    {
        data0.store(mem);
        data1.store(mem + storage_type0::size());
    }
    template <typename Flags> Vc_INTRINSIC void store(bool *mem, Flags f) const
    {
        data0.store(mem, f);
        data1.store(mem + storage_type0::size(), f);
    }

    Vc_INTRINSIC Vc_PURE bool operator==(const simd_mask_array &rhs) const
    {
        return data0 == rhs.data0 && data1 == rhs.data1;
    }
    Vc_INTRINSIC Vc_PURE bool operator!=(const simd_mask_array &rhs) const
    {
        return data0 != rhs.data0 || data1 != rhs.data1;
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator!() const
    {
        return {!data0, !data1};
    }

    Vc_INTRINSIC simd_mask_array &operator&=(const simd_mask_array &rhs)
    {
        data0 &= rhs.data0;
        data1 &= rhs.data1;
        return *this;
    }
    Vc_INTRINSIC simd_mask_array &operator|=(const simd_mask_array &rhs)
    {
        data0 |= rhs.data0;
        data1 |= rhs.data1;
        return *this;
    }
    Vc_INTRINSIC simd_mask_array &operator^=(const simd_mask_array &rhs)
    {
        data0 ^= rhs.data0;
        data1 ^= rhs.data1;
        return *this;
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator&(const simd_mask_array &rhs) const
    {
        return {data0 & rhs.data0, data1 & rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator|(const simd_mask_array &rhs) const
    {
        return {data0 | rhs.data0, data1 | rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator^(const simd_mask_array &rhs) const
    {
        return {data0 ^ rhs.data0, data1 ^ rhs.data1};
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array operator&&(const simd_mask_array &rhs) const
    {
        return {data0 && rhs.data0, data1 && rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE simd_mask_array operator||(const simd_mask_array &rhs) const
    {
        return {data0 || rhs.data0, data1 || rhs.data1};
    }

    Vc_INTRINSIC Vc_PURE bool isFull() const { return data0.isFull() && data1.isFull(); }
    Vc_INTRINSIC Vc_PURE bool isNotEmpty() const { return data0.isNotEmpty() && data1.isNotEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isEmpty() const { return data0.isEmpty() && data1.isEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isMix() const { return data0.isMix() || data1.isMix(); }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
    Vc_INTRINSIC Vc_PURE operator bool() const { return isFull(); }
#endif

    Vc_INTRINSIC Vc_PURE bool operator[](size_t index) const {
        auto alias = reinterpret_cast<const vectorentry_type *>(&data0);
        return alias[index];
    }

    Vc_INTRINSIC Vc_PURE int count() const { return data0.count() + data1.count(); }

    Vc_INTRINSIC Vc_PURE int firstOne() const {
        if (data0.isEmpty()) {
            return data1.firstOne() + storage_type0::size();
        }
        return data0.firstOne();
    }

    /// \internal
    Vc_INTRINSIC simd_mask_array(storage_type0 &&x, storage_type1 &&y)
        : data0(std::move(x)), data1(std::move(y))
    {
    }

private:
    storage_type0 data0;
    storage_type1 data1;
};

#if 0
{
    Vc_ALWAYS_INLINE explicit simd_mask_array(VectorSpecialInitializerZero::ZEnum) : d(false) {}
    Vc_ALWAYS_INLINE explicit simd_mask_array(VectorSpecialInitializerOne::OEnum) : d(true) {}
    Vc_ALWAYS_INLINE simd_mask_array(bool x) : d(x) {}

    // default copy ctor/operator
    simd_mask_array(const simd_mask_array &) = default;
    simd_mask_array(simd_mask_array &&) = default;
    template <typename U> simd_mask_array(const simd_mask_array<U, N> &x) : d(x.d)
    {
    }
    simd_mask_array &operator=(const simd_mask_array &) = default;

    Vc_ALWAYS_INLINE Vc_PURE bool isFull() const { return d.isFull(); }
    Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const { return d.isEmpty(); }

#define VC_COMPARE_IMPL(op)                                                                        \
    Vc_ALWAYS_INLINE Vc_PURE bool operator op(const simd_mask_array &x) const                      \
    {                                                                                              \
        return d.apply([](bool l, bool r) { return l && r; },                                      \
                       [](mask_type l, mask_type r) { return l op r; },                            \
                       x.d);                                                                       \
    }
    VC_ALL_COMPARES(VC_COMPARE_IMPL)
#undef VC_COMPARE_IMPL

    bool operator[](std::size_t i) const {
        const auto m = d.cbegin();
        return m[i / mask_type::Size][i % mask_type::Size];
    }

    simd_mask_array operator!() const
    {
        simd_mask_array r;
        r.d.assign(d, &mask_type::operator!);
        return r;
    }

    unsigned int count() const
    {
        return d.count();
    }

//private:
    storage_type d;

    friend const decltype(d) & simd_mask_array_data(const simd_mask_array &x) { return x.d; }
    friend decltype(d) & simd_mask_array_data(simd_mask_array &x) { return x.d; }
    friend decltype(std::move(d)) simd_mask_array_data(simd_mask_array &&x) { return std::move(x.d); }
};
#endif

}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_SIMD_MASK_ARRAY_H
