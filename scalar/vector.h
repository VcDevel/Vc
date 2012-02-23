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

#ifndef SCALAR_VECTOR_H
#define SCALAR_VECTOR_H

#include <assert.h>
#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
#include <float.h>
#endif

#include "../common/memoryfwd.h"
#include "macros.h"
#include "types.h"
#include "vectorbase.h"
#include "mask.h"
#include "writemaskedvector.h"

namespace Vc
{
namespace Scalar
{
    enum { VectorAlignment = 4 };

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    friend class WriteMaskedVector<T>;
    protected:
        T m_data;
    public:
        typedef Vc::Memory<Vector<T>, 1> Memory;
        typedef T EntryType;
        typedef Vector<unsigned int> IndexType;
        typedef Scalar::Mask<1u> Mask;

        T &data() { return m_data; }
        T data() const { return m_data; }

        enum { Size = 1 };
        inline Vector() {}
        inline Vector(VectorSpecialInitializerZero::ZEnum) : m_data(0) {}
        inline Vector(VectorSpecialInitializerOne::OEnum) : m_data(1) {}
        inline Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : m_data(0) {}

        static inline Vector Zero() { Vector r; r.m_data = 0; return r; }
        static inline Vector IndexesFromZero() { return Zero(); }

        template<typename OtherT> explicit inline Vector(const Vector<OtherT> *a) : m_data(static_cast<T>(a->data())) {}
        template<typename OtherT> explicit inline Vector(const Vector<OtherT> &x) : m_data(static_cast<T>(x.data())) {}
        inline Vector(T x) : m_data(x) {}
        explicit inline Vector(const T *x) : m_data(x[0]) {}
        template<typename A> inline Vector(const T *x, A) : m_data(x[0]) {}
        template<typename Other> inline Vector(const Other *x) : m_data(x[0]) {}
        template<typename Other, typename A> inline Vector(const Other *x, A) : m_data(x[0]) {}

        template<typename OtherT> inline void expand(Vector<OtherT> *x) const { x->data() = static_cast<OtherT>(m_data); }

        inline void setZero() { m_data = 0; }
        inline void setZero(Mask k) { if (k) m_data = 0; }

        template<typename Other> inline void load(const Other *mem) { m_data = mem[0]; }
        template<typename Other, typename A> inline void load(const Other *mem, A) { m_data = mem[0]; }
        template<typename Other> inline void load(const Other *mem, Mask m) { if (m.data()) m_data = mem[0]; }

        inline void load(const T *mem) { m_data = mem[0]; }
        template<typename A> inline void load(const T *mem, A) { m_data = mem[0]; }
        inline void load(const T *mem, Mask m) { if (m.data()) m_data = mem[0]; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void store(T *mem) const { mem[0] = m_data; }
        inline void store(T *mem, Mask m) const { if (m.data()) mem[0] = m_data; }
        template<typename A> inline void store(T *mem, A) const { store(mem); }
        template<typename A> inline void store(T *mem, Mask m, A) const { store(mem, m); }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return *this; }
        inline const Vector<T> badc() const { return *this; }
        inline const Vector<T> aaaa() const { return *this; }
        inline const Vector<T> bbbb() const { return *this; }
        inline const Vector<T> cccc() const { return *this; }
        inline const Vector<T> dddd() const { return *this; }
        inline const Vector<T> dbac() const { return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> inline Vector(const T *array, const IndexT *indexes) : m_data(array[indexes[0]]) {}
        template<typename IndexT> inline Vector(const T *array, Vector<IndexT> indexes) : m_data(array[indexes[0]]) {}
        template<typename IndexT> inline Vector(const T *array, IndexT indexes, Mask m) : m_data(m.data() ? array[indexes[0]] : 0) {}
        template<typename S1, typename IT> inline Vector(const S1 *array, const T S1::* member1, IT indexes, Mask mask = true)
            : m_data(mask.data() ? (&array[indexes[0]])->*(member1) : 0) {}
        template<typename S1, typename S2, typename IT> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, IT indexes, Mask mask = true)
            : m_data(mask.data() ? array[indexes[0]].*(member1).*(member2) : 0) {}
        template<typename S1, typename IT1, typename IT2> inline Vector(const S1 *array, const EntryType *const S1::* ptrMember1,
                IT1 outerIndex, IT2 innerIndex, Mask mask = true)
            : m_data(mask.data() ? (array[outerIndex[0]].*(ptrMember1))[innerIndex[0]] : 0) {}

        template<typename IT> inline void gather(const T *array, IT indexes, Mask mask = true)
            { if (mask.data()) m_data = array[indexes[0]]; }
        template<typename S1, typename IT> inline void gather(const S1 *array, const T S1::* member1, IT indexes, Mask mask = true)
            { if (mask.data()) m_data = (&array[indexes[0]])->*(member1); }
        template<typename S1, typename S2, typename IT> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, IT indexes, Mask mask = true)
            { if (mask.data()) m_data = array[indexes[0]].*(member1).*(member2); }
        template<typename S1, typename IT1, typename IT2> inline void gather(const S1 *array, const EntryType *const S1::* ptrMember1,
                IT1 outerIndex, IT2 innerIndex, Mask mask = true)
            { if (mask.data()) m_data = (array[outerIndex[0]].*(ptrMember1))[innerIndex[0]]; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        inline void scatter(T *array, const Vector<unsigned int> &indexes, Mask m = true) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S1> inline void scatter(S1 *array, T S1::* member, const Vector<unsigned int> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member) = m_data;
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1, T S2::* member2,
                const Vector<unsigned int> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member1).*(member2) = m_data;
        }

        inline void scatter(T *array, const Vector<unsigned short> &indexes, Mask m = true) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S1> inline void scatter(S1 *array, T S1::* member, const Vector<unsigned short> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member) = m_data;
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1, T S2::* member2,
                const Vector<unsigned short> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member1).*(member2) = m_data;
        }

        //prefix
        inline Vector &operator++() { ++m_data; return *this; }
        //postfix
        inline Vector operator++(int) { return m_data++; }

        inline T &operator[](int index) {
            assert(index == 0); if(index) {}
            return m_data;
        }

        inline T operator[](int index) const {
            assert(index == 0); if(index) {}
            return m_data;
        }

        inline Vector operator~() const { return ~m_data; }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const { return -m_data; }

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) { m_data symbol##= x.m_data; return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(m_data symbol x.m_data); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
        OP(/, div)
        OP(%, rem)
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const { return m_data symbol x.m_data; } \
        inline Mask operator symbol(const T &x) const { return m_data symbol x; }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) {
            m_data *= factor;
            m_data += summand;
        }

        inline Vector<T> multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) const {
            return Vector<T>( m_data * factor.m_data + summand.m_data );
        }

        inline void assign(const Vector<T> &v, const Mask &m) {
          if (m.data()) m_data = v.m_data;
        }

        template<typename V2> inline V2 staticCast() const { return static_cast<typename V2::EntryType>(m_data); }
        template<typename V2> inline V2 reinterpretCast() const { return reinterpret_cast<typename V2::EntryType>(m_data); }

        inline WriteMaskedVector<T> operator()(Mask m) { return WriteMaskedVector<T>(this, m); }

        inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            if (!m1.data() && m2.data()) {
                m_data = v2.m_data;
                m1 = true;
                m2 = false;
                return true;
            }
            return m1;
        }

        inline EntryType min() const { return m_data; }
        inline EntryType max() const { return m_data; }
        inline EntryType product() const { return m_data; }
        inline EntryType sum() const { return m_data; }
        inline EntryType min(Mask) const { return m_data; }
        inline EntryType max(Mask) const { return m_data; }
        inline EntryType product(Mask) const { return m_data; }
        inline EntryType sum(Mask m) const { if (m) return m_data; return static_cast<EntryType>(0); }

        Vector sorted() const { return *this; }

        template<typename F> void callWithValuesSorted(F &f) {
            f(m_data);
        }
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T1, typename T2> inline Vector<T2> operator+ (T1 x, const Vector<T2> &v) { return v.operator+(x); }
template<typename T1, typename T2> inline Vector<T2> operator* (T1 x, const Vector<T2> &v) { return v.operator*(x); }
template<typename T1, typename T2> inline Vector<T2> operator- (T1 x, const Vector<T2> &v) { return Vector<T2>(x) - v; }
template<typename T1, typename T2> inline Vector<T2> operator/ (T1 x, const Vector<T2> &v) { return Vector<T2>(x) / v; }
template<typename T1, typename T2> inline Mask<1u>   operator< (T1 x, const Vector<T2> &v) { return Vector<T2>(x) <  v; }
template<typename T1, typename T2> inline Mask<1u>   operator<=(T1 x, const Vector<T2> &v) { return Vector<T2>(x) <= v; }
template<typename T1, typename T2> inline Mask<1u>   operator> (T1 x, const Vector<T2> &v) { return Vector<T2>(x) >  v; }
template<typename T1, typename T2> inline Mask<1u>   operator>=(T1 x, const Vector<T2> &v) { return Vector<T2>(x) >= v; }
template<typename T1, typename T2> inline Mask<1u>   operator==(T1 x, const Vector<T2> &v) { return Vector<T2>(x) == v; }
template<typename T1, typename T2> inline Mask<1u>   operator!=(T1 x, const Vector<T2> &v) { return Vector<T2>(x) != v; }

#define PARENT_DATA (static_cast<Vector<T> *>(this)->m_data)
#define PARENT_DATA_CONST (static_cast<const Vector<T> *>(this)->m_data)
#define OP_IMPL(symbol) \
  template<> inline Vector<T> &VectorBase<T, Vector<T> >::operator symbol##=(const Vector<T> &x) { PARENT_DATA symbol##= x.m_data; return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T> VectorBase<T, Vector<T> >::operator symbol(const Vector<T> &x) const { return Vector<T>(PARENT_DATA_CONST symbol x.m_data); }

#define T int
  OP_IMPL(&)
  OP_IMPL(|)
  OP_IMPL(^)
  OP_IMPL(<<)
  OP_IMPL(>>)
#undef T
#define T unsigned int
  OP_IMPL(&)
  OP_IMPL(|)
  OP_IMPL(^)
  OP_IMPL(<<)
  OP_IMPL(>>)
#undef T
#define T short
  OP_IMPL(&)
  OP_IMPL(|)
  OP_IMPL(^)
  OP_IMPL(<<)
  OP_IMPL(>>)
#undef T
#define T unsigned short
  OP_IMPL(&)
  OP_IMPL(|)
  OP_IMPL(^)
  OP_IMPL(<<)
  OP_IMPL(>>)
#undef T
#undef OP_IMPL
#undef PARENT_DATA_CONST
#undef PARENT_DATA

#ifdef _MSC_VER
  template<typename T> static inline void forceToRegisters(const Vector<T> &) {
  }
#else
  template<typename T> static inline void forceToRegisters(const Vector<T> &x01) {
      __asm__ __volatile__(""::"r"(x01.data()));
  }
  template<> inline void forceToRegisters(const Vector<float> &x01) {
      __asm__ __volatile__(""::"x"(x01.data()));
  }
  template<> inline void forceToRegisters(const Vector<double> &x01) {
      __asm__ __volatile__(""::"x"(x01.data()));
  }
#endif
  template<typename T1, typename T2> static inline void forceToRegisters(
      const Vector<T1> &x01, const Vector<T2> &x02) {
      forceToRegisters(x01);
      forceToRegisters(x02);
  }
  template<typename T1, typename T2, typename T3> static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &, const Vector<T3>  &) {}
  template<typename T1, typename T2, typename T3, typename T4> static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &,
        const Vector<T9>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14> static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15> static inline void forceToRegisters(
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
    typename T14, typename T15, typename T16> static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &,
        const Vector<T15> &, const Vector<T16> &) {}

} // namespace Scalar
} // namespace Vc

#include "memory.h"
#include "math.h"
#include "undomacros.h"

#endif // SCALAR_VECTOR_H
