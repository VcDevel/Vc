/*  This file is part of the Vc library.

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

#ifndef SCALAR_VECTOR_H
#define SCALAR_VECTOR_H

#include <assert.h>
#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
#include <float.h>
#endif

#include "../common/memoryfwd.h"
#include "../common/loadstoreflags.h"
#include "types.h"
#include "mask.h"
#include "writemaskedvector.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    enum VectorAlignmentEnum { VectorAlignment = 4 };

template<typename T>
class Vector
{
    friend class WriteMaskedVector<T>;
    public:
        typedef typename DetermineEntryType<T>::Type EntryType;
    protected:
        EntryType m_data;
    public:
        typedef Vc::Memory<Vector<T>, 1> Memory;
        typedef Vector<unsigned int> IndexType;
        typedef Scalar::Mask<T> Mask;
        typedef Vector<T> AsArg;

        Vc_ALWAYS_INLINE EntryType &data() { return m_data; }
        Vc_ALWAYS_INLINE EntryType data() const { return m_data; }

        static constexpr size_t Size = 1;
        enum Constants {
            MemoryAlignment = alignof(EntryType)
        };

        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        Vc_ALWAYS_INLINE Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        Vc_ALWAYS_INLINE Vector(VectorSpecialInitializerZero::ZEnum) : m_data(0) {}
        Vc_ALWAYS_INLINE Vector(VectorSpecialInitializerOne::OEnum) : m_data(1) {}
        Vc_ALWAYS_INLINE Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : m_data(0) {}
        static Vc_ALWAYS_INLINE Vector Zero() { Vector r; r.m_data = 0; return r; }
        static Vc_ALWAYS_INLINE Vector One() { Vector r; r.m_data = 1; return r; }
        static Vc_ALWAYS_INLINE Vector IndexesFromZero() { return Zero(); }
        static Vc_INTRINSIC_L Vector Random() Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // copy
        Vc_INTRINSIC Vector(const Vector &x) = default;
        Vc_INTRINSIC Vector &operator=(const Vector &v) { m_data = v.data(); return *this; }

        // implict conversion from compatible Vector<U>
        template<typename U> Vc_INTRINSIC Vector(VC_ALIGNED_PARAMETER(Vector<U>) x,
                typename std::enable_if<is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
            : m_data(static_cast<EntryType>(x.data())) {}

        // static_cast from the remaining Vector<U>
        template<typename U> Vc_INTRINSIC explicit Vector(VC_ALIGNED_PARAMETER(Vector<U>) x,
                typename std::enable_if<!is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
            : m_data(static_cast<EntryType>(x.data())) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vc_INTRINSIC Vector(EntryType a) : m_data(a) {}
        template <typename U>
        Vc_INTRINSIC Vector(U a,
                            typename std::enable_if<std::is_same<U, int>::value &&
                                                        !std::is_same<U, EntryType>::value,
                                                    void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit Vc_INTRINSIC Vector(const EntryType *x) { load(x); }
        template<typename Flags = AlignedT> explicit Vc_INTRINSIC Vector(const EntryType * x, Flags flags = Flags())
        {
            load(x, flags);
        }
        template<typename OtherT, typename Flags = AlignedT> explicit Vc_INTRINSIC Vector(const OtherT *x, Flags flags = Flags(), typename std::enable_if<std::is_fundamental<OtherT>::value, void *>::type = nullptr)
        {
            load(x, flags);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand 1 float_v to 2 double_v                 XXX rationale? remove it for release? XXX
        template<typename OtherT> Vc_ALWAYS_INLINE void expand(Vector<OtherT> *x) const { x->data() = static_cast<OtherT>(m_data); }
        template<typename OtherT> explicit Vc_ALWAYS_INLINE Vector(const Vector<OtherT> *a) : m_data(static_cast<EntryType>(a->data())) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        Vc_ALWAYS_INLINE void setZero() { m_data = 0; }
        Vc_ALWAYS_INLINE void setZero(Mask k) { if (k) m_data = 0; }

        Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setQnan(Mask m) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        Vc_INTRINSIC void load(const EntryType *mem) { m_data = mem[0]; }
        template<typename Flags = AlignedT> Vc_INTRINSIC
            void load(const EntryType *mem, Flags) { m_data = mem[0]; }
        template<typename OtherT, typename Flags = AlignedT> Vc_INTRINSIC
            void load(const OtherT    *mem, Flags = Flags()) { m_data = mem[0]; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC void store(T2 *mem, Flags = Flags()) const { mem[0] = m_data; }
        template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC void store(T2 *mem, Mask mask, Flags = Flags()) const { if (mask.data()) mem[0] = m_data; }
        // the following store overloads are here to support classes that have a cast operator to EntryType.
        // Without this overload GCC complains about not finding a matching store function.
        Vc_INTRINSIC void store(EntryType *mem) const { store<EntryType, AlignedT>(mem); }
        template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Flags flags) const { store<EntryType, Flags>(mem, flags); }
        Vc_INTRINSIC void store(EntryType *mem, Mask mask) const { store<EntryType, AlignedT>(mem, mask); }
        template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Mask mask, Flags flags) const { store<EntryType, Flags>(mem, mask, flags); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        Vc_INTRINSIC const Vector<T> &abcd() const { return *this; }
        Vc_INTRINSIC const Vector<T>  cdab() const { return *this; }
        Vc_INTRINSIC const Vector<T>  badc() const { return *this; }
        Vc_INTRINSIC const Vector<T>  aaaa() const { return *this; }
        Vc_INTRINSIC const Vector<T>  bbbb() const { return *this; }
        Vc_INTRINSIC const Vector<T>  cccc() const { return *this; }
        Vc_INTRINSIC const Vector<T>  dddd() const { return *this; }
        Vc_INTRINSIC const Vector<T>  bcad() const { return *this; }
        Vc_INTRINSIC const Vector<T>  bcda() const { return *this; }
        Vc_INTRINSIC const Vector<T>  dabc() const { return *this; }
        Vc_INTRINSIC const Vector<T>  acbd() const { return *this; }
        Vc_INTRINSIC const Vector<T>  dbca() const { return *this; }
        Vc_INTRINSIC const Vector<T>  dcba() const { return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> Vc_ALWAYS_INLINE Vector(const EntryType *array, const IndexT *indexes) : m_data(array[indexes[0]]) {}
        template<typename IndexT> Vc_ALWAYS_INLINE Vector(const EntryType *array, Vector<IndexT> indexes) : m_data(array[indexes[0]]) {}
        template<typename IndexT> Vc_ALWAYS_INLINE Vector(const EntryType *array, IndexT indexes, Mask m) : m_data(m.data() ? array[indexes[0]] : 0) {}
        template<typename S1, typename IT> Vc_ALWAYS_INLINE Vector(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask = Mask(true))
            : m_data(mask.data() ? (&array[indexes[0]])->*(member1) : 0) {}
        template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vector(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, IT indexes, Mask mask = Mask(true))
            : m_data(mask.data() ? array[indexes[0]].*(member1).*(member2) : 0) {}
        template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE Vector(const S1 *array, const EntryType *const S1::* ptrMember1,
                IT1 outerIndex, IT2 innerIndex, Mask mask = Mask(true))
            : m_data(mask.data() ? (array[outerIndex[0]].*(ptrMember1))[innerIndex[0]] : 0) {}

        template<typename IT> Vc_ALWAYS_INLINE void gather(const EntryType *array, IT indexes, Mask mask = Mask(true))
            { if (mask.data()) m_data = array[indexes[0]]; }
        template<typename S1, typename IT> Vc_ALWAYS_INLINE void gather(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask = Mask(true))
            { if (mask.data()) m_data = (&array[indexes[0]])->*(member1); }
        template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE void gather(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, IT indexes, Mask mask = Mask(true))
            { if (mask.data()) m_data = array[indexes[0]].*(member1).*(member2); }
        template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE void gather(const S1 *array, const EntryType *const S1::* ptrMember1,
                IT1 outerIndex, IT2 innerIndex, Mask mask = Mask(true))
            { if (mask.data()) m_data = (array[outerIndex[0]].*(ptrMember1))[innerIndex[0]]; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        Vc_ALWAYS_INLINE void scatter(EntryType *array, const Vector<unsigned int> &indexes, Mask m = Mask(true)) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S1> Vc_ALWAYS_INLINE void scatter(S1 *array, EntryType S1::* member, const Vector<unsigned int> &indexes, Mask m = Mask(true)) const {
            if (m.data()) array[indexes[0]].*(member) = m_data;
        }
        template<typename S1, typename S2> Vc_ALWAYS_INLINE void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2,
                const Vector<unsigned int> &indexes, Mask m = Mask(true)) const {
            if (m.data()) array[indexes[0]].*(member1).*(member2) = m_data;
        }

        Vc_ALWAYS_INLINE void scatter(EntryType *array, const Vector<unsigned short> &indexes, Mask m = Mask(true)) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S1> Vc_ALWAYS_INLINE void scatter(S1 *array, EntryType S1::* member, const Vector<unsigned short> &indexes, Mask m = Mask(true)) const {
            if (m.data()) array[indexes[0]].*(member) = m_data;
        }
        template<typename S1, typename S2> Vc_ALWAYS_INLINE void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2,
                const Vector<unsigned short> &indexes, Mask m = Mask(true)) const {
            if (m.data()) array[indexes[0]].*(member1).*(member2) = m_data;
        }

        //prefix
        Vc_ALWAYS_INLINE Vector &operator++() { ++m_data; return *this; }
        Vc_ALWAYS_INLINE Vector &operator--() { --m_data; return *this; }
        //postfix
        Vc_ALWAYS_INLINE Vector operator++(int) { return m_data++; }
        Vc_ALWAYS_INLINE Vector operator--(int) { return m_data--; }

        Vc_ALWAYS_INLINE EntryType &operator[](size_t index) {
            assert(index == 0); if(index) {}
            return m_data;
        }

        Vc_ALWAYS_INLINE EntryType operator[](size_t index) const {
            assert(index == 0); if(index) {}
            return m_data;
        }

        Vc_ALWAYS_INLINE Vector operator~() const { return Vector(~m_data); }
        Vc_ALWAYS_INLINE Vector<typename NegateTypeHelper<T>::Type> operator-() const { return Vector<typename NegateTypeHelper<T>::Type>(-m_data); }
        Vc_INTRINSIC Vector Vc_PURE operator+() const { return *this; }

#define OPshift(symbol) \
        Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector &x) { m_data symbol##= x.m_data; return *this; } \
        Vc_ALWAYS_INLINE Vc_PURE Vector operator symbol(const Vector &x) const { return Vector<T>(m_data symbol x.m_data); }
        VC_ALL_SHIFTS(OPshift)
#undef OPshift

#define OP(symbol) \
        Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector &x) { m_data symbol##= x.m_data; return *this; } \
        Vc_ALWAYS_INLINE Vc_PURE Vector operator symbol(const Vector &x) const { return Vector(m_data symbol x.m_data); }
        VC_ALL_ARITHMETICS(OP)
        VC_ALL_BINARY(OP)
#undef OP

#define OPcmp(symbol) \
        Vc_ALWAYS_INLINE Vc_PURE Mask operator symbol(const Vector &x) const { return Mask(m_data symbol x.m_data); }
        VC_ALL_COMPARES(OPcmp)
#undef OPcmp

        Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_PURE_R Vc_INTRINSIC_R;

        Vc_ALWAYS_INLINE void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand) {
            m_data = m_data * factor.data() + summand.data();
        }

        Vc_ALWAYS_INLINE void assign(const Vector<T> &v, const Mask &m) {
          if (m.data()) m_data = v.m_data;
        }

        template<typename V2> Vc_ALWAYS_INLINE V2 staticCast() const { return V2(static_cast<typename V2::EntryType>(m_data)); }
        template<typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const {
            typedef typename V2::EntryType AliasT2 Vc_MAY_ALIAS;
            return V2(*reinterpret_cast<const AliasT2 *>(&m_data));
        }

        Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(Mask m) { return WriteMaskedVector<T>(this, m); }

        Vc_ALWAYS_INLINE bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            if (!m1.data() && m2.data()) {
                m_data = v2.m_data;
                m1 = true;
                m2 = false;
                return true;
            }
            return m1;
        }

        Vc_ALWAYS_INLINE EntryType min() const { return m_data; }
        Vc_ALWAYS_INLINE EntryType max() const { return m_data; }
        Vc_ALWAYS_INLINE EntryType product() const { return m_data; }
        Vc_ALWAYS_INLINE EntryType sum() const { return m_data; }
        Vc_ALWAYS_INLINE Vector partialSum() const { return *this; }
        Vc_ALWAYS_INLINE EntryType min(Mask) const { return m_data; }
        Vc_ALWAYS_INLINE EntryType max(Mask) const { return m_data; }
        Vc_ALWAYS_INLINE EntryType product(Mask) const { return m_data; }
        Vc_ALWAYS_INLINE EntryType sum(Mask m) const { if (m) return m_data; return static_cast<EntryType>(0); }

        Vc_INTRINSIC Vector shifted(int amount, Vector shiftIn) const {
            VC_ASSERT(amount >= -1 && amount <= 1);
            return amount == 0 ? *this : shiftIn;
        }
        Vc_INTRINSIC Vector shifted(int amount) const { return amount == 0 ? *this : Zero(); }
        Vc_INTRINSIC Vector rotated(int) const { return *this; }
        Vector sorted() const { return *this; }

#ifdef VC_NO_MOVE_CTOR
        template<typename F> Vc_INTRINSIC void call(const F &f) const {
            f(m_data);
        }
        template<typename F> Vc_INTRINSIC void call(const F &f, Mask mask) const {
            if (mask) {
                f(m_data);
            }
        }
        template<typename F> Vc_INTRINSIC Vector apply(const F &f) const {
            return Vector(f(m_data));
        }
        template<typename F> Vc_INTRINSIC Vector apply(const F &f, Mask mask) const {
            if (mask) {
                return Vector(f(m_data));
            } else {
                return *this;
            }
        }
#endif
        template<typename F> void callWithValuesSorted(F VC_RR_ f) {
            f(m_data);
        }

        template<typename F> Vc_INTRINSIC void call(F VC_RR_ f) const {
            f(m_data);
        }

        template<typename F> Vc_INTRINSIC void call(F VC_RR_ f, Mask mask) const {
            if (mask) {
                f(m_data);
            }
        }

        template<typename F> Vc_INTRINSIC Vector apply(F VC_RR_ f) const {
            return Vector(f(m_data));
        }

        template<typename F> Vc_INTRINSIC Vector apply(F VC_RR_ f, Mask mask) const {
            if (mask) {
                return Vector(f(m_data));
            } else {
                return *this;
            }
        }

        template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
            m_data = f(0);
        }
        Vc_INTRINSIC void fill(EntryType (&f)()) {
            m_data = f();
        }

        Vc_INTRINSIC_L Vector copySign(Vector reference) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector exponent() const Vc_INTRINSIC_R;
};
template<typename T> constexpr size_t Vector<T>::Size;

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;
typedef double_v::Mask double_m;
typedef float_v::Mask float_m;
typedef int_v::Mask int_m;
typedef uint_v::Mask uint_m;
typedef short_v::Mask short_m;
typedef ushort_v::Mask ushort_m;

template<typename T> class SwizzledVector : public Vector<T> {};

#ifdef _MSC_VER
  template<typename T> static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T> &) {
  }
#else
  template<typename T> static Vc_ALWAYS_INLINE void forceToRegisters(const Vector<T> &x01) {
      __asm__ __volatile__(""::"r"(x01.data()));
  }
  template<> Vc_ALWAYS_INLINE void forceToRegisters(const Vector<float> &x01) {
      __asm__ __volatile__(""::"x"(x01.data()));
  }
  template<> Vc_ALWAYS_INLINE void forceToRegisters(const Vector<double> &x01) {
      __asm__ __volatile__(""::"x"(x01.data()));
  }
#endif
  template<typename T1, typename T2> static Vc_ALWAYS_INLINE void forceToRegisters(
      const Vector<T1> &x01, const Vector<T2> &x02) {
      forceToRegisters(x01);
      forceToRegisters(x02);
  }
  template<typename T1, typename T2, typename T3> static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &, const Vector<T3>  &) {}
  template<typename T1, typename T2, typename T3, typename T4> static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &,
        const Vector<T9>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13>
    static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14> static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15> static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &,
        const Vector<T15> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16> static Vc_ALWAYS_INLINE void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &,
        const Vector<T15> &, const Vector<T16> &) {}

Vc_IMPL_NAMESPACE_END

#include "vector.tcc"
#include "math.h"
#include "undomacros.h"

#endif // SCALAR_VECTOR_H
