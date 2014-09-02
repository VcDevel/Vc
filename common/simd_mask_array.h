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
/// \addtogroup simdarray
/// @{

template <typename T, std::size_t N, typename VectorType_> class simd_mask_array<T, N, VectorType_, N>
{
public:
    using VectorType = VectorType_;
    using vector_type = VectorType;
    using mask_type = typename vector_type::Mask;
    using storage_type = mask_type;

    friend storage_type &internal_data(simd_mask_array &m) { return m.data; }
    friend const storage_type &internal_data(const simd_mask_array &m) { return m.data; }

    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename mask_type::VectorEntryType;
    using vectorentry_reference = vectorentry_type &;
    using value_type = typename mask_type::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using Vector = simdarray<T, N, VectorType, N>;

    FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    simd_mask_array() = default;

    // broadcasts
    Vc_INTRINSIC explicit simd_mask_array(VectorSpecialInitializerOne::OEnum one) : data(one) {}
    Vc_INTRINSIC explicit simd_mask_array(VectorSpecialInitializerZero::ZEnum zero) : data(zero) {}
    Vc_INTRINSIC explicit simd_mask_array(bool b) : data(b) {}
    Vc_INTRINSIC static simd_mask_array Zero() { return {storage_type::Zero()}; }
    Vc_INTRINSIC static simd_mask_array One() { return {storage_type::One()}; }

    // conversion (casts)
    template <typename U, typename V>
    Vc_INTRINSIC_L simd_mask_array(const simd_mask_array<U, N, V> &x,
                                   enable_if<N == V::size()> = nullarg) Vc_INTRINSIC_R;
    template <typename U, typename V>
    Vc_INTRINSIC_L simd_mask_array(const simd_mask_array<U, N, V> &x,
                                   enable_if<(N > V::size() && N <= 2 * V::size())> = nullarg)
        Vc_INTRINSIC_R;
    template <typename U, typename V>
    Vc_INTRINSIC_L simd_mask_array(const simd_mask_array<U, N, V> &x,
                                   enable_if<(N > 2 * V::size() && N <= 4 * V::size())> = nullarg)
        Vc_INTRINSIC_R;

    // conversion from any Segment object (could be simd_mask_array or Mask<T>)
    template <typename M, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC_L simd_mask_array(
        Common::Segment<M, Pieces, Index> &&x,
        enable_if<Traits::simd_vector_size<M>::value == Size * Pieces> = nullarg) Vc_INTRINSIC_R;

    // conversion from Mask<T>
    template <typename M>
    Vc_INTRINSIC_L simd_mask_array(
        M k,
        enable_if<(Traits::is_simd_mask<M>::value && !Traits::is_simd_mask_array<M>::value &&
                   Traits::simd_vector_size<M>::value == Size)> = nullarg) Vc_INTRINSIC_R;

    // load/store (from/to bool arrays)
    template <typename Flags = DefaultLoadTag>
    Vc_INTRINSIC explicit simd_mask_array(const bool *mem, Flags f = Flags())
        : data(mem, f)
    {
    }

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

    // compares
    Vc_INTRINSIC Vc_PURE bool operator==(const simd_mask_array &rhs) const
    {
        return data == rhs.data;
    }
    Vc_INTRINSIC Vc_PURE bool operator!=(const simd_mask_array &rhs) const
    {
        return data != rhs.data;
    }

    // inversion
    Vc_INTRINSIC Vc_PURE simd_mask_array operator!() const
    {
        return {!data};
    }

    // binary operators
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

    Vc_INTRINSIC Vc_PURE int shiftMask() const { return data.shiftMask(); }

    Vc_INTRINSIC Vc_PURE int toInt() const { return data.toInt(); }

    Vc_INTRINSIC Vc_PURE vectorentry_reference operator[](size_t index)
    {
        return data[index];
    }
    Vc_INTRINSIC Vc_PURE bool operator[](size_t index) const { return data[index]; }

    Vc_INTRINSIC Vc_PURE int count() const { return data.count(); }

    /**
     * Returns the index of the first one in the mask.
     *
     * The return value is undefined if the mask is empty.
     */
    Vc_INTRINSIC Vc_PURE int firstOne() const { return data.firstOne(); }

    template <typename G> static Vc_INTRINSIC simd_mask_array generate(const G &gen)
    {
        return {mask_type::generate(gen)};
    }

    Vc_INTRINSIC Vc_PURE simd_mask_array shifted(int amount) const
    {
        return {data.shifted(amount)};
    }

    /// \internal execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC simd_mask_array fromOperation(Op op, Args &&... args)
    {
        simd_mask_array r;
        op(r.data, Common::actual_value(op, std::forward<Args>(args))...);
        return r;
    }

    /// \internal
    Vc_INTRINSIC simd_mask_array(mask_type &&x) : data(std::move(x)) {}

private:
    storage_type data;
};

template <typename T, std::size_t N, typename VectorType> constexpr std::size_t simd_mask_array<T, N, VectorType, N>::Size;

template <typename T, std::size_t N, typename VectorType, std::size_t> class simd_mask_array
{
    static constexpr std::size_t N0 = Common::nextPowerOfTwo(N - N / 2);

    using Split = Common::Split<N0>;

public:
    using storage_type0 = simd_mask_array<T, N0>;
    using storage_type1 = simd_mask_array<T, N - N0>;
    static_assert(storage_type0::size() == N0, "");

    using vector_type = VectorType;

    friend storage_type0 &internal_data0(simd_mask_array &m) { return m.data0; }
    friend storage_type1 &internal_data1(simd_mask_array &m) { return m.data1; }
    friend const storage_type0 &internal_data0(const simd_mask_array &m) { return m.data0; }
    friend const storage_type1 &internal_data1(const simd_mask_array &m) { return m.data1; }

    using mask_type = simd_mask_array;
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename storage_type0::VectorEntryType;
    using vectorentry_reference = vectorentry_type &;
    /* FIXME:
    static_assert(std::is_same<vectorentry_type, typename storage_type1::VectorEntryType>::value,
                  "incompatible mask types combined: this will break operator[]");
     */
    using value_type = typename storage_type0::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using Vector = simdarray<T, N, VectorType, VectorType::Size>;

    FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    simd_mask_array() = default;

    // default copy ctor/operator
    simd_mask_array(const simd_mask_array &) = default;
    simd_mask_array(simd_mask_array &&) = default;
    simd_mask_array &operator=(const simd_mask_array &) = default;
    simd_mask_array &operator=(simd_mask_array &&) = default;

    // implicit conversion from simd_mask_array with same N
    template <typename U, typename V>
    Vc_INTRINSIC simd_mask_array(const simd_mask_array<U, N, V> &rhs)
        : data0(Split::lo(rhs)), data1(Split::hi(rhs))
    {
    }

    // conversion from any Segment object (could be simd_mask_array or Mask<T>)
    template <typename M, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC simd_mask_array(
        Common::Segment<M, Pieces, Index> &&rhs,
        enable_if<Traits::simd_vector_size<M>::value == Size * Pieces> = nullarg)
        : data0(Split::lo(rhs)), data1(Split::hi(rhs))
    {
    }

    // conversion from Mask<T>
    template <typename M>
    Vc_INTRINSIC simd_mask_array(
        M k,
        enable_if<(Traits::is_simd_mask<M>::value && !Traits::is_simd_mask_array<M>::value &&
                   Traits::simd_vector_size<M>::value == Size)> = nullarg)
        : data0(Split::lo(k)), data1(Split::hi(k))
    {
    }

    Vc_INTRINSIC explicit simd_mask_array(VectorSpecialInitializerOne::OEnum one)
        : data0(one), data1(one)
    {
    }
    Vc_INTRINSIC explicit simd_mask_array(VectorSpecialInitializerZero::ZEnum zero)
        : data0(zero), data1(zero)
    {
    }
    Vc_INTRINSIC explicit simd_mask_array(bool b) : data0(b), data1(b) {}

    Vc_INTRINSIC static simd_mask_array Zero() { return {storage_type0::Zero(), storage_type1::Zero()}; }
    Vc_INTRINSIC static simd_mask_array One() { return {storage_type0::One(), storage_type1::One()}; }

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
    Vc_INTRINSIC Vc_PURE bool isNotEmpty() const { return data0.isNotEmpty() || data1.isNotEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isEmpty() const { return data0.isEmpty() && data1.isEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isMix() const { return !isFull() && !isEmpty(); }

    Vc_INTRINSIC Vc_PURE int toInt() const
    {
        return data0.toInt() | (data1.toInt() << data0.size());
    }

    Vc_INTRINSIC Vc_PURE vectorentry_reference operator[](size_t index) {
        auto alias = reinterpret_cast<vectorentry_type *>(&data0);
        return alias[index];
    }
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

    template <typename G> static Vc_INTRINSIC simd_mask_array generate(const G &gen)
    {
        return {storage_type0::generate(gen),
                storage_type1::generate([&](std::size_t i) { return gen(i + N0); })};
    }

    inline Vc_PURE simd_mask_array shifted(int amount) const
    {
        if (VC_IS_UNLIKELY(amount == 0)) {
            return *this;
        }
        simd_mask_array r{};
        if (amount < 0) {
            for (int i = 0; i < int(Size) + amount; ++i) {
                r[i - amount] = operator[](i);
            }
        } else {
            for (int i = 0; i < int(Size) - amount; ++i) {
                r[i] = operator[](i + amount);
            }
        }
        return r;
    }

    /// \internal execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC simd_mask_array fromOperation(Op op, Args &&... args)
    {
        simd_mask_array r = {
            storage_type0::fromOperation(op, Split::lo(args)...),  // no forward here - it
                                                                   // could move and thus
                                                                   // break the next line
            storage_type1::fromOperation(op, Split::lo(std::forward<Args>(args))...)};
        return r;
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
template <typename T, std::size_t N, typename VectorType, std::size_t M> constexpr std::size_t simd_mask_array<T, N, VectorType, M>::Size;

/// @}

}  // namespace Vc

#include "undomacros.h"

// XXX: this include should be in <Vc/vector.h>. But at least clang 3.4 then fails to compile the
// code. Not sure yet what is going on, but it looks a lot like a bug in clang.
#include "simd_cast_caller.tcc"

#endif // VC_COMMON_SIMD_MASK_ARRAY_H
