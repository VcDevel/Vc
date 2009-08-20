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

#ifndef SIMPLE_VECTOR_H
#define SIMPLE_VECTOR_H

#include <assert.h>
#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
#include <float.h>
#endif

#ifndef ALIGN
# ifdef __GNUC__
#  define ALIGN(n) __attribute__((aligned(n)))
# else
#  define ALIGN(n) __declspec(align(n))
# endif
#endif

namespace Simple
{
    namespace { template<typename T1> void UNUSED_PARAM1( const T1 & ) {} }

    enum { VectorAlignment = 4 };

    class VectorAlignedBase {};

    template<typename T> class Vector;

#define PARENT_DATA (static_cast<Parent *>(this)->m_data)
#define PARENT_DATA_CONST (static_cast<const Parent *>(this)->m_data)
    template<typename T, typename Parent> struct VectorBase {};

#define OP_DECL(symbol) \
        inline Vector<T> &operator symbol##=(const Vector<T> &x); \
        inline Vector<T> operator symbol(const Vector<T> &x) const;
    template<typename Parent> struct VectorBase<int, Parent>
    {
#define T int
        OP_DECL(|)
        OP_DECL(&)
        OP_DECL(^)
        OP_DECL(<<)
        OP_DECL(>>)
#undef T
    };
    template<typename Parent> struct VectorBase<unsigned int, Parent>
    {
#define T unsigned int
        OP_DECL(|)
        OP_DECL(&)
        OP_DECL(^)
        OP_DECL(<<)
        OP_DECL(>>)
#undef T
    };
    template<typename Parent> struct VectorBase<short, Parent>
    {
#define T short
        OP_DECL(|)
        OP_DECL(&)
        OP_DECL(^)
        OP_DECL(<<)
        OP_DECL(>>)
#undef T
    };
    template<typename Parent> struct VectorBase<unsigned short, Parent>
    {
#define T unsigned short
        OP_DECL(|)
        OP_DECL(&)
        OP_DECL(^)
        OP_DECL(<<)
        OP_DECL(>>)
#undef T
    };
#undef PARENT_DATA
#undef PARENT_DATA_CONST

namespace VectorSpecialInitializerZero { enum ZEnum { Zero }; }
namespace VectorSpecialInitializerOne { enum OEnum { One }; }
namespace VectorSpecialInitializerRandom { enum REnum { Random }; }
namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

template<unsigned int VectorSize = 1>
class Mask
{
    public:
        inline Mask() {}
        inline Mask(bool b) : m(b) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : m(false) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : m(true) {}
        inline Mask(const Mask<VectorSize> *a) : m(a[0].m) {}

        inline void expand(Mask *x) { x[0].m = m; }

        inline bool operator==(const Mask &rhs) const { return m == rhs.m; }
        inline bool operator!=(const Mask &rhs) const { return m != rhs.m; }
        inline Mask operator&&(const Mask &rhs) const { return m && rhs.m; }
        inline Mask operator||(const Mask &rhs) const { return m || rhs.m; }
        inline Mask operator!() const { return !m; }
        inline Mask operator&(const Mask &rhs) const { return m && rhs.m; }
        inline Mask operator|(const Mask &rhs) const { return m || rhs.m; }
        inline Mask &operator&=(const Mask &rhs) { m &= rhs.m; return *this; }
        inline Mask &operator|=(const Mask &rhs) { m |= rhs.m; return *this; }
        inline bool isFull () const { return  m; }
        inline bool isEmpty() const { return !m; }
        inline bool isMix  () const { return false; }

        inline bool data () const { return m; }
        inline bool dataI() const { return m; }
        inline bool dataD() const { return m; }

        inline operator bool() const { return isFull(); }

        template<unsigned int OtherSize>
        inline Mask cast() const { return *this; }

        inline bool operator[](int) const { return m; }

        Mask<VectorSize * 2> combine(Mask other) const;

    private:
        bool m;
};

/**
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
 * \param it   The iterator variable. For example "int i".
 * \param mask The mask to iterate over. You can also just write a vector operation that returns a
 *             mask.
 */
template<typename F>
inline void foreach_bit(const Mask<1u> &mask, F func) {
    if (mask) {
        func(0);
    }
}

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef bool Mask;
    public:
        //prefix
        inline Vector<T> &operator++() { if (mask) ++vec->m_data; return *vec; }
        inline Vector<T> &operator--() { if (mask) --vec->m_data; return *vec; }
        //postfix
        inline Vector<T> operator++(int) { if (mask) return vec->m_data++; return *vec; }
        inline Vector<T> operator--(int) { if (mask) return vec->m_data--; return *vec; }

        inline Vector<T> &operator+=(Vector<T> x) { if (mask) vec->m_data += x.m_data; return *vec; }
        inline Vector<T> &operator-=(Vector<T> x) { if (mask) vec->m_data -= x.m_data; return *vec; }
        inline Vector<T> &operator*=(Vector<T> x) { if (mask) vec->m_data *= x.m_data; return *vec; }
        inline Vector<T> &operator/=(Vector<T> x) { if (mask) vec->m_data /= x.m_data; return *vec; }

        inline Vector<T> &operator=(Vector<T> x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k) {}
        Vector<T> *vec;
        Mask mask;
};

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    friend class WriteMaskedVector<T>;
    protected:
        T m_data;
    public:
        typedef T EntryType;
        typedef Vector<unsigned int> IndexType;
        typedef Simple::Mask<1u> Mask;

        T &data() { return m_data; }
        T data() const { return m_data; }

        enum { Size = 1 };
        inline Vector() {}
        inline Vector(VectorSpecialInitializerZero::ZEnum) : m_data(0) {}
        inline Vector(VectorSpecialInitializerOne::OEnum) : m_data(1) {}
        inline Vector(VectorSpecialInitializerRandom::REnum) { makeRandom(); }
        inline Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : m_data(0) {}

        template<typename OtherT> explicit inline Vector(const Vector<OtherT> *a) : m_data(static_cast<T>(a->data())) {}
        template<typename OtherT> explicit inline Vector(const Vector<OtherT> &x) : m_data(static_cast<T>(x.data())) {}
        inline Vector(T x) : m_data(x) {}
        template<typename Other> inline Vector(const Other *x) : m_data(x[0]) {}

        template<typename OtherT> inline void expand(Vector<OtherT> *x) const { x->data() = static_cast<OtherT>(m_data); }

        inline void makeZero() { m_data = 0; }
        inline void makeZero(Mask k) { if (k) m_data = 0; }
        inline void makeRandom() { m_data = std::rand(); }
        inline void makeRandom(Mask k) { if (k) m_data = std::rand(); }

        template<typename Other> inline void load(const Other *mem) { m_data = mem[0]; }
        template<typename Other> inline void load(const Other *mem, Mask m) { if (m.data()) m_data = mem[0]; }
        static inline Vector loadUnaligned(const T *mem) { return Vector(mem[0]); }
        template<typename Other> inline void store(Other *mem) const { mem[0] = m_data; }
        template<typename Other> inline void store(Other *mem, Mask m) const { if (m.data()) mem[0] = m_data; }
        template<typename Other> inline void storeStreaming(Other *mem) const { mem[0] = m_data; }
        template<typename Other> inline void storeStreaming(Other *mem, Mask m) const { if (m.data()) mem[0] = m_data; }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return *this; }
        inline const Vector<T> badc() const { return *this; }
        inline const Vector<T> aaaa() const { return *this; }
        inline const Vector<T> bbbb() const { return *this; }
        inline const Vector<T> cccc() const { return *this; }
        inline const Vector<T> dddd() const { return *this; }
        inline const Vector<T> dbac() const { return *this; }

        inline Vector(const T *array, const Vector<unsigned int> &indexes) : m_data(array[indexes[0]]) {}
        inline Vector(const T *array, const Vector<unsigned int> &indexes, Mask m) : m_data(m.data() ? array[indexes[0]] : 0) {}
        inline void gather(const T *array, const Vector<unsigned int> &indexes) { m_data = array[indexes[0]]; }
        inline void gather(const T *array, const Vector<unsigned int> &indexes, Mask m) { if (m.data()) m_data = array[indexes[0]]; }

        inline Vector(const T *array, const Vector<unsigned short> &indexes) : m_data(array[indexes[0]]) {}
        inline Vector(const T *array, const Vector<unsigned short> &indexes, Mask m) : m_data(m.data() ? array[indexes[0]] : 0) {}
        inline void gather(const T *array, const Vector<unsigned short> &indexes) { m_data = array[indexes[0]]; }
        inline void gather(const T *array, const Vector<unsigned short> &indexes, Mask m) { if (m.data()) m_data = array[indexes[0]]; }

        template<typename S> inline Vector(const S *array, const T S::* member1,
                const Vector<unsigned int> &indexes, Mask mask = true)
            : m_data(mask.data() ? (&array[indexes[0]])->*(member1) : 0) {}

        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes, Mask mask = true)
            : m_data(mask.data() ? array[indexes[0]].*(member1).*(member2) : 0) {}

        template<typename S> inline void gather(const S *array, const T S::* member1,
                const Vector<unsigned int> &indexes, Mask mask = true) {
            if (mask.data()) m_data = (&array[indexes[0]])->*(member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes, Mask mask = true) {
            if (mask.data()) m_data = array[indexes[0]].*(member1).*(member2);
        }

        inline void scatter(T *array, const Vector<unsigned int> &indexes, Mask m = true) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S> inline void scatter(S *array, T S::* member, const Vector<unsigned int> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member) = m_data;
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1, T S2::* member2,
                const Vector<unsigned int> &indexes, Mask m = true) const {
            if (m.data()) array[indexes[0]].*(member1).*(member2) = m_data;
        }

        template<typename S> inline Vector(const S *array, const T S::* member1,
                const Vector<unsigned short> &indexes, Mask mask = true)
            : m_data(mask.data() ? (&array[indexes[0]])->*(member1) : 0) {}

        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes, Mask mask = true)
            : m_data(mask.data() ? array[indexes[0]].*(member1).*(member2) : 0) {}

        template<typename S> inline void gather(const S *array, const T S::* member1,
                const Vector<unsigned short> &indexes, Mask mask = true) {
            if (mask.data()) m_data = (&array[indexes[0]])->*(member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes, Mask mask = true) {
            if (mask.data()) m_data = array[indexes[0]].*(member1).*(member2);
        }

        inline void scatter(T *array, const Vector<unsigned short> &indexes, Mask m = true) const { if (m.data()) array[indexes[0]] = m_data; }
        template<typename S> inline void scatter(S *array, T S::* member, const Vector<unsigned short> &indexes, Mask m = true) const {
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
        inline void increment(Mask mask) { if (mask.data()) ++m_data; }
        inline void decrement(Mask mask) { if (mask.data()) --m_data; }

        inline T operator[](int index) const {
            assert(index == 0); UNUSED_PARAM1(index);
            return m_data;
        }

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(std::fun(m_data)); } \
        inline Vector &fun##_eq() { m_data = std::fun(m_data); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

        inline Vector operator~() const { return ~m_data; }

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
        inline Mask operator symbol(const Vector<T> &x) const { return m_data symbol x.m_data; }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp

        inline Vector mulHigh(const Vector<T> &factor) const {
            //STATIC_ASSERT(typeid(T) == typeid(int) || typeid(T) == typeid(unsigned int), mulHigh_only_supported_for_32bit_integers);
            //STATIC_ASSERT(typeid(T) == typeid(unsigned int), mulHigh_only_supported_for_32bit_integers);
            unsigned long long int x = m_data;
            //int64_t x = m_data;
            x *= factor;
            return Vector<T>(x >> 32);
        }

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

        inline T max() const { return m_data; }

        template<typename T2> inline Vector<T2> staticCast() const { return static_cast<T2>(m_data); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return reinterpret_cast<T2>(m_data); }

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

        template<typename F> void callWithValuesSorted(F &f) {
            f(m_data);
        }
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const T &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> operator-(const T &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const T &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline Mask<1u>  operator< (const T &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline Mask<1u>  operator<=(const T &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline Mask<1u>  operator> (const T &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline Mask<1u>  operator>=(const T &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline Mask<1u>  operator==(const T &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline Mask<1u>  operator!=(const T &x, const Vector<T> &v) { return Vector<T>(x) != v; }

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
#undef ALIGN
#undef PARENT_DATA_CONST
#undef PARENT_DATA

  template<typename T> static inline Simple::Vector<T> min  (const Simple::Vector<T> &x, const T &y) { return std::min( x.data(), y ); }
  template<typename T> static inline Simple::Vector<T> max  (const Simple::Vector<T> &x, const T &y) { return std::max( x.data(), y ); }
  template<typename T> static inline Simple::Vector<T> min  (const T &x, const Simple::Vector<T> &y) { return std::min( x, y.data() ); }
  template<typename T> static inline Simple::Vector<T> max  (const T &x, const Simple::Vector<T> &y) { return std::max( x, y.data() ); }
  template<typename T> static inline Simple::Vector<T> min  (const Simple::Vector<T> &x, const Simple::Vector<T> &y) { return std::min( x.data(), y.data() ); }
  template<typename T> static inline Simple::Vector<T> max  (const Simple::Vector<T> &x, const Simple::Vector<T> &y) { return std::max( x.data(), y.data() ); }
  template<typename T> static inline Simple::Vector<T> sqrt (const Simple::Vector<T> &x) { return std::sqrt( x.data() ); }
  template<typename T> static inline Simple::Vector<T> rsqrt(const Simple::Vector<T> &x) { const T one = 1; return one / std::sqrt( x.data() ); }
  template<typename T> static inline Simple::Vector<T> abs  (const Simple::Vector<T> &x) { return std::abs( x.data() ); }
  template<typename T> static inline Simple::Vector<T> sin  (const Simple::Vector<T> &x) { return std::sin( x.data() ); }
  template<typename T> static inline Simple::Vector<T> cos  (const Simple::Vector<T> &x) { return std::cos( x.data() ); }
  template<typename T> static inline Simple::Vector<T> log  (const Simple::Vector<T> &x) { return std::log( x.data() ); }
  template<typename T> static inline Simple::Vector<T> log10(const Simple::Vector<T> &x) { return std::log10( x.data() ); }
  template<typename T> static inline Simple::Vector<T> atan (const Simple::Vector<T> &x) { return std::atan( x.data() ); }
  template<typename T> static inline Simple::Vector<T> atan2(const Simple::Vector<T> &x, const Simple::Vector<T> &y) { return std::atan2( x.data(), y.data() ); }
  template<typename T> static inline Simple::Vector<T> reciprocal(const Simple::Vector<T> &x) { const T one = 1; return one / x.data(); }

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif
  template<typename T> static inline bool isfinite(const Simple::Vector<T> &x) { return
#ifdef _MSC_VER
      !!_finite(x.data());
#elif defined(__INTEL_COMPILER)
      ::isfinite(x.data());
#else
      std::isfinite(x.data());
#endif
  }
  template<typename T> static inline bool isnan(const Simple::Vector<T> &x) { return
#ifdef _MSC_VER
      !!_isnan(x.data());
#elif defined(__INTEL_COMPILER)
      ::isnan(x.data());
#else
      std::isnan(x.data());
#endif
  }

  template<typename T> static inline void forceToRegisters(const Vector<T> &) {}
  template<typename T1, typename T2> static inline void forceToRegisters(
      const Vector<T1> &, const Vector<T2> &) {}
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
} // namespace Simple

#endif // SIMPLE_VECTOR_H
