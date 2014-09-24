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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_NEON_MASK_H_
#define VC_NEON_MASK_H_

#include "../common/maskentry.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace NEON
{
template <typename T> class Mask
{
public:
    using EntryType = bool;
    using VectorEntryType = Common::MaskBool<sizeof(T)>;
    using Vector = NEON::Vector<T>;
    using VectorType = typename Vector::VectorType;
    using EntryReference = VectorEntryType &;

    static constexpr size_t Size = Vector::Size;
    static constexpr std::size_t size() { return Size; }

    FREE_STORE_OPERATORS_ALIGNED(alignof(VectorType))

    Vc_INTRINSIC_L Mask() Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Mask(VectorType x) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L explicit Mask(VectorSpecialInitializerZero::ZEnum) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L explicit Mask(VectorSpecialInitializerOne::OEnum) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L explicit Mask(bool b) Vc_INTRINSIC_R;

    template <typename U>
    using enable_if_implicitly_convertible = enable_if<
        (Traits::is_simd_mask<U>::value && !Traits::is_simd_mask_array<U>::value &&
         is_implicit_cast_allowed_mask<Traits::entry_type_of<typename Traits::decay<U>::Vector>,
                                       T>::value)>;
    template <typename U>
    using enable_if_explicitly_convertible = enable_if<
        (Traits::is_simd_mask_array<U>::value ||
         (Traits::is_simd_mask<U>::value &&
          !is_implicit_cast_allowed_mask<Traits::entry_type_of<typename Traits::decay<U>::Vector>,
                                         T>::value))>;

    // implicit cast
    template <typename U>
    Vc_INTRINSIC_L Mask(U &&rhs, enable_if_implicitly_convertible<U> = nullarg) Vc_INTRINSIC_R;

    // explicit cast, implemented via simd_cast (in avx/simd_cast_caller.h)
    template <typename U>
    Vc_INTRINSIC_L explicit Mask(U &&rhs,
                                 enable_if_explicitly_convertible<U> = nullarg) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L explicit Mask(const bool *mem) Vc_INTRINSIC_R;
    template <typename Flags> Vc_INTRINSIC_L explicit Mask(const bool *mem, Flags f) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L void load(const bool *mem) Vc_INTRINSIC_R;
    template <typename Flags> Vc_INTRINSIC_L void load(const bool *mem, Flags) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L void store(bool *) const Vc_INTRINSIC_R;
    template <typename Flags> Vc_INTRINSIC_L void store(bool *mem, Flags) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L Vc_PURE_L bool operator==(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L bool operator!=(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator!() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Mask &operator&=(const Mask &rhs) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Mask &operator|=(const Mask &rhs) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Mask &operator^=(const Mask &rhs) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator&(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator|(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator^(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator&&(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L Mask operator||(const Mask &rhs) const Vc_INTRINSIC_R Vc_PURE_R;

    Vc_INTRINSIC_L Vc_PURE_L bool isFull() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L bool isNotEmpty() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L bool isEmpty() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L bool isMix() const Vc_INTRINSIC_R Vc_PURE_R;

    Vc_INTRINSIC_L Vc_PURE_L int shiftMask() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L int toInt() const Vc_INTRINSIC_R Vc_PURE_R;

    Vc_INTRINSIC_L EntryReference operator[](size_t index) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vc_PURE_L bool operator[](size_t index) const Vc_INTRINSIC_R Vc_PURE_R;

    Vc_INTRINSIC_L Vc_PURE_L unsigned int count() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L unsigned int firstOne() const Vc_INTRINSIC_R Vc_PURE_R;

    // internal:
    friend Vc_INTRINSIC VectorType internal_data(const Mask &k) { return k.data; }
    friend Vc_INTRINSIC VectorType &internal_data(Mask &k) { return k.data; }

private:
    VectorType data; // TODO: Mask type member corresponding to Vector<T>
};
}  // namespace NEON
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_NEON_MASK_H_

// vim: foldmethod=marker
