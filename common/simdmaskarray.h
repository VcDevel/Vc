/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_SIMDMASKARRAY_H_
#define VC_COMMON_SIMDMASKARRAY_H_

#include <type_traits>
#include <array>
#include "simdarrayhelper.h"
#include "utility.h"
#include "maskentry.h"

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
/// \addtogroup SimdArray
/// @{

template <typename T, std::size_t N, typename VectorType_>
class alignas(
    ((Common::nextPowerOfTwo(N) * (sizeof(VectorType_) / VectorType_::size()) - 1) & 127) +
    1) SimdMaskArray<T, N, VectorType_, N>
{
public:
    using VectorType = VectorType_;
    using vector_type = VectorType;
    using mask_type = typename vector_type::Mask;
    using storage_type = mask_type;

    friend storage_type &internal_data(SimdMaskArray &m) { return m.data; }
    friend const storage_type &internal_data(const SimdMaskArray &m) { return m.data; }

    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static constexpr std::size_t MemoryAlignment = storage_type::MemoryAlignment;
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename mask_type::VectorEntryType;
    using vectorentry_reference = vectorentry_type &;
    using value_type = typename mask_type::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using EntryReference = typename mask_type::EntryReference;
    using Vector = SimdArray<T, N, VectorType, N>;

    Vc_FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    SimdMaskArray() = default;

    // broadcasts
    Vc_INTRINSIC explicit SimdMaskArray(VectorSpecialInitializerOne one) : data(one) {}
    Vc_INTRINSIC explicit SimdMaskArray(VectorSpecialInitializerZero zero) : data(zero) {}
    Vc_INTRINSIC explicit SimdMaskArray(bool b) : data(b) {}
    Vc_INTRINSIC static SimdMaskArray Zero() { return {storage_type::Zero()}; }
    Vc_INTRINSIC static SimdMaskArray One() { return {storage_type::One()}; }

    // conversion (casts)
    template <typename U, typename V>
    Vc_INTRINSIC_L SimdMaskArray(const SimdMaskArray<U, N, V> &x,
                                   enable_if<N == V::size()> = nullarg) Vc_INTRINSIC_R;
    template <typename U, typename V>
    Vc_INTRINSIC_L SimdMaskArray(const SimdMaskArray<U, N, V> &x,
                                   enable_if<(N > V::size() && N <= 2 * V::size())> = nullarg)
        Vc_INTRINSIC_R;
    template <typename U, typename V>
    Vc_INTRINSIC_L SimdMaskArray(const SimdMaskArray<U, N, V> &x,
                                   enable_if<(N > 2 * V::size() && N <= 4 * V::size())> = nullarg)
        Vc_INTRINSIC_R;

    // conversion from any Segment object (could be SimdMaskArray or Mask<T>)
    template <typename M, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC_L SimdMaskArray(
        Common::Segment<M, Pieces, Index> &&x,
        enable_if<Traits::simd_vector_size<M>::value == Size * Pieces> = nullarg) Vc_INTRINSIC_R;

    // conversion from Mask<T>
    template <typename M>
    Vc_INTRINSIC_L SimdMaskArray(
        M k,
        enable_if<(Traits::is_simd_mask<M>::value && !Traits::isSimdMaskArray<M>::value &&
                   Traits::simd_vector_size<M>::value == Size)> = nullarg) Vc_INTRINSIC_R;

    // implicit conversion to Mask<U, AnyAbi> for if Mask<U, AnyAbi>::size() == N
    template <typename M,
              typename = enable_if<Traits::is_simd_mask<M>::value &&
                                   !Traits::isSimdMaskArray<M>::value && M::size() == N>>
    operator M() const
    {
        return simd_cast<M>(*this);
    }

    // load/store (from/to bool arrays)
    template <typename Flags = DefaultLoadTag>
    Vc_INTRINSIC explicit SimdMaskArray(const bool *mem, Flags f = Flags())
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
    Vc_INTRINSIC Vc_PURE bool operator==(const SimdMaskArray &rhs) const
    {
        return data == rhs.data;
    }
    Vc_INTRINSIC Vc_PURE bool operator!=(const SimdMaskArray &rhs) const
    {
        return data != rhs.data;
    }

    // inversion
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator!() const
    {
        return {!data};
    }

    // binary operators
    Vc_INTRINSIC SimdMaskArray &operator&=(const SimdMaskArray &rhs)
    {
        data &= rhs.data;
        return *this;
    }
    Vc_INTRINSIC SimdMaskArray &operator|=(const SimdMaskArray &rhs)
    {
        data |= rhs.data;
        return *this;
    }
    Vc_INTRINSIC SimdMaskArray &operator^=(const SimdMaskArray &rhs)
    {
        data ^= rhs.data;
        return *this;
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray operator&(const SimdMaskArray &rhs) const
    {
        return {data & rhs.data};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator|(const SimdMaskArray &rhs) const
    {
        return {data | rhs.data};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator^(const SimdMaskArray &rhs) const
    {
        return {data ^ rhs.data};
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray operator&&(const SimdMaskArray &rhs) const
    {
        return {data && rhs.data};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator||(const SimdMaskArray &rhs) const
    {
        return {data || rhs.data};
    }

    Vc_INTRINSIC Vc_PURE bool isFull() const { return data.isFull(); }
    Vc_INTRINSIC Vc_PURE bool isNotEmpty() const { return data.isNotEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isEmpty() const { return data.isEmpty(); }
    Vc_INTRINSIC Vc_PURE bool isMix() const { return data.isMix(); }

    Vc_INTRINSIC Vc_PURE int shiftMask() const { return data.shiftMask(); }

    Vc_INTRINSIC Vc_PURE int toInt() const { return data.toInt(); }

    Vc_INTRINSIC Vc_PURE EntryReference operator[](size_t index)
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

    template <typename G> static Vc_INTRINSIC SimdMaskArray generate(const G &gen)
    {
        return {mask_type::generate(gen)};
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray shifted(int amount) const
    {
        return {data.shifted(amount)};
    }

    /// \internal execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC SimdMaskArray fromOperation(Op op, Args &&... args)
    {
        SimdMaskArray r;
        Common::unpackArgumentsAuto(op, &r.data, std::forward<Args>(args)...);
        return r;
    }

    /// \internal
    Vc_INTRINSIC SimdMaskArray(mask_type &&x) : data(std::move(x)) {}

    ///\internal Called indirectly from operator[]
    void setEntry(size_t index, bool x) { data.setEntry(index, x); }

private:
    storage_type data;
};

template <typename T, std::size_t N, typename VectorType> constexpr std::size_t SimdMaskArray<T, N, VectorType, N>::Size;
template <typename T, std::size_t N, typename VectorType>
constexpr std::size_t SimdMaskArray<T, N, VectorType, N>::MemoryAlignment;

template <typename T, std::size_t N, typename VectorType, std::size_t>
class alignas(
    ((Common::nextPowerOfTwo(N) * (sizeof(VectorType) / VectorType::size()) - 1) & 127) +
    1) SimdMaskArray
{
    static constexpr std::size_t N0 = Common::nextPowerOfTwo(N - N / 2);

    using Split = Common::Split<N0>;

public:
    using storage_type0 = SimdMaskArray<T, N0>;
    using storage_type1 = SimdMaskArray<T, N - N0>;
    static_assert(storage_type0::size() == N0, "");

    using vector_type = VectorType;

    friend storage_type0 &internal_data0(SimdMaskArray &m) { return m.data0; }
    friend storage_type1 &internal_data1(SimdMaskArray &m) { return m.data1; }
    friend const storage_type0 &internal_data0(const SimdMaskArray &m) { return m.data0; }
    friend const storage_type1 &internal_data1(const SimdMaskArray &m) { return m.data1; }

    using mask_type = SimdMaskArray;
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static constexpr std::size_t MemoryAlignment =
        storage_type0::MemoryAlignment > storage_type1::MemoryAlignment
            ? storage_type0::MemoryAlignment
            : storage_type1::MemoryAlignment;
    static_assert(Size == mask_type::Size, "size mismatch");

    using vectorentry_type = typename storage_type0::VectorEntryType;
    using vectorentry_reference = vectorentry_type &;
    using value_type = typename storage_type0::EntryType;
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using EntryReference = typename std::conditional<
        std::is_same<typename storage_type0::EntryReference,
                     typename storage_type1::EntryReference>::value,
        typename storage_type0::EntryReference, Common::MaskEntry<SimdMaskArray>>::type;
    using Vector = SimdArray<T, N, VectorType, VectorType::Size>;

    Vc_FREE_STORE_OPERATORS_ALIGNED(alignof(mask_type))

    // zero init
    SimdMaskArray() = default;

    // default copy ctor/operator
    SimdMaskArray(const SimdMaskArray &) = default;
    SimdMaskArray(SimdMaskArray &&) = default;
    SimdMaskArray &operator=(const SimdMaskArray &) = default;
    SimdMaskArray &operator=(SimdMaskArray &&) = default;

    // implicit conversion from SimdMaskArray with same N
    template <typename U, typename V>
    Vc_INTRINSIC SimdMaskArray(const SimdMaskArray<U, N, V> &rhs)
        : data0(Split::lo(rhs)), data1(Split::hi(rhs))
    {
    }

    // conversion from any Segment object (could be SimdMaskArray or Mask<T>)
    template <typename M, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC SimdMaskArray(
        Common::Segment<M, Pieces, Index> &&rhs,
        enable_if<Traits::simd_vector_size<M>::value == Size * Pieces> = nullarg)
        : data0(Split::lo(rhs)), data1(Split::hi(rhs))
    {
    }

    // conversion from Mask<T>
    template <typename M>
    Vc_INTRINSIC SimdMaskArray(
        M k,
        enable_if<(Traits::is_simd_mask<M>::value && !Traits::isSimdMaskArray<M>::value &&
                   Traits::simd_vector_size<M>::value == Size)> = nullarg)
        : data0(Split::lo(k)), data1(Split::hi(k))
    {
    }

    // implicit conversion to Mask<U, AnyAbi> for if Mask<U, AnyAbi>::size() == N
    template <typename M,
              typename = enable_if<Traits::is_simd_mask<M>::value &&
                                   !Traits::isSimdMaskArray<M>::value && M::size() == N>>
    operator M() const
    {
        return simd_cast<M>(*this);
    }

    Vc_INTRINSIC explicit SimdMaskArray(VectorSpecialInitializerOne one)
        : data0(one), data1(one)
    {
    }
    Vc_INTRINSIC explicit SimdMaskArray(VectorSpecialInitializerZero zero)
        : data0(zero), data1(zero)
    {
    }
    Vc_INTRINSIC explicit SimdMaskArray(bool b) : data0(b), data1(b) {}

    Vc_INTRINSIC static SimdMaskArray Zero() { return {storage_type0::Zero(), storage_type1::Zero()}; }
    Vc_INTRINSIC static SimdMaskArray One() { return {storage_type0::One(), storage_type1::One()}; }

    template <typename Flags = DefaultLoadTag>
    Vc_INTRINSIC explicit SimdMaskArray(const bool *mem, Flags f = Flags())
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

    Vc_INTRINSIC Vc_PURE bool operator==(const SimdMaskArray &rhs) const
    {
        return data0 == rhs.data0 && data1 == rhs.data1;
    }
    Vc_INTRINSIC Vc_PURE bool operator!=(const SimdMaskArray &rhs) const
    {
        return data0 != rhs.data0 || data1 != rhs.data1;
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray operator!() const
    {
        return {!data0, !data1};
    }

    Vc_INTRINSIC SimdMaskArray &operator&=(const SimdMaskArray &rhs)
    {
        data0 &= rhs.data0;
        data1 &= rhs.data1;
        return *this;
    }
    Vc_INTRINSIC SimdMaskArray &operator|=(const SimdMaskArray &rhs)
    {
        data0 |= rhs.data0;
        data1 |= rhs.data1;
        return *this;
    }
    Vc_INTRINSIC SimdMaskArray &operator^=(const SimdMaskArray &rhs)
    {
        data0 ^= rhs.data0;
        data1 ^= rhs.data1;
        return *this;
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray operator&(const SimdMaskArray &rhs) const
    {
        return {data0 & rhs.data0, data1 & rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator|(const SimdMaskArray &rhs) const
    {
        return {data0 | rhs.data0, data1 | rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator^(const SimdMaskArray &rhs) const
    {
        return {data0 ^ rhs.data0, data1 ^ rhs.data1};
    }

    Vc_INTRINSIC Vc_PURE SimdMaskArray operator&&(const SimdMaskArray &rhs) const
    {
        return {data0 && rhs.data0, data1 && rhs.data1};
    }
    Vc_INTRINSIC Vc_PURE SimdMaskArray operator||(const SimdMaskArray &rhs) const
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

private:
    template <typename R>
    R subscript_impl(
        size_t index,
        enable_if<std::is_same<R, typename storage_type0::EntryReference>::value> =
            nullarg)
    {
        if (index < storage_type0::size()) {
            return data0[index];
        } else {
            return data1[index - storage_type0::size()];
        }
    }
    template <typename R>
    R subscript_impl(
        size_t index,
        enable_if<!std::is_same<R, typename storage_type0::EntryReference>::value> =
            nullarg)
    {
        return {*this, index};
    }

public:
    ///\internal Called indirectly from operator[]
    void setEntry(size_t index, bool x)
    {
        if (index < data0.size()) {
            data0.setEntry(index, x);
        } else {
            data1.setEntry(index - data0.size(), x);
        }
    }

    Vc_INTRINSIC Vc_PURE EntryReference operator[](size_t index) {
        return subscript_impl<EntryReference>(index);
    }
    Vc_INTRINSIC Vc_PURE bool operator[](size_t index) const {
        if (index < storage_type0::size()) {
            return data0[index];
        } else {
            return data1[index - storage_type0::size()];
        }
    }

    Vc_INTRINSIC Vc_PURE int count() const { return data0.count() + data1.count(); }

    Vc_INTRINSIC Vc_PURE int firstOne() const {
        if (data0.isEmpty()) {
            return data1.firstOne() + storage_type0::size();
        }
        return data0.firstOne();
    }

    template <typename G> static Vc_INTRINSIC SimdMaskArray generate(const G &gen)
    {
        return {storage_type0::generate(gen),
                storage_type1::generate([&](std::size_t i) { return gen(i + N0); })};
    }

    inline Vc_PURE SimdMaskArray shifted(int amount) const
    {
        if (Vc_IS_UNLIKELY(amount == 0)) {
            return *this;
        }
        SimdMaskArray r{};
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
    static Vc_INTRINSIC SimdMaskArray fromOperation(Op op, Args &&... args)
    {
        SimdMaskArray r = {
            storage_type0::fromOperation(op, Split::lo(args)...),  // no forward here - it
                                                                   // could move and thus
                                                                   // break the next line
            storage_type1::fromOperation(op, Split::hi(std::forward<Args>(args))...)};
        return r;
    }

    /// \internal
    Vc_INTRINSIC SimdMaskArray(storage_type0 &&x, storage_type1 &&y)
        : data0(std::move(x)), data1(std::move(y))
    {
    }

private:
    storage_type0 data0;
    storage_type1 data1;
};
template <typename T, std::size_t N, typename VectorType, std::size_t M> constexpr std::size_t SimdMaskArray<T, N, VectorType, M>::Size;
template <typename T, std::size_t N, typename VectorType, std::size_t M>
constexpr std::size_t SimdMaskArray<T, N, VectorType, M>::MemoryAlignment;

/// @}

}  // namespace Vc

// XXX: this include should be in <Vc/vector.h>. But at least clang 3.4 then fails to compile the
// code. Not sure yet what is going on, but it looks a lot like a bug in clang.
#include "simd_cast_caller.tcc"

#endif // VC_COMMON_SIMDMASKARRAY_H_
