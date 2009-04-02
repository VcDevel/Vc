/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef SIMPLE_VECTOR_H
#define SIMPLE_VECTOR_H

#include <assert.h>
#include <algorithm>
#include <cmath>

namespace Simple
{
    namespace
    {
        template<typename T1> void UNUSED_PARAM1( const T1 & ) {}
    }

    enum { VectorAlignment = 8 };

    template<typename T> class Vector;

#define PARENT_DATA (static_cast<Parent *>(this)->m_data)
#define PARENT_DATA_CONST (static_cast<const Parent *>(this)->m_data)
    template<typename T, typename Parent> struct VectorBase;

    template<typename Parent>
    struct VectorBase<float, Parent>
    {
        //operator float &() { return PARENT_DATA; }
        //operator float() const { return PARENT_DATA_CONST; }
        enum Upconvert { UpconvertNone = 0x00   /* no conversion      */ };
        Vector<int> toInt() const;
        Vector<unsigned int> toUInt() const;
    };

    template<typename Parent>
    struct VectorBase<double, Parent>
    {
        //operator double &() { return PARENT_DATA; }
        //operator double() const { return PARENT_DATA_CONST; }
        enum Upconvert { UpconvertNone = 0x00   /* no conversion      */ };
    };

    template<typename Parent>
    struct VectorBase<int, Parent>
    {
        //operator int &() { return PARENT_DATA; }
        //operator int() const { return PARENT_DATA_CONST; }

        Vector<float> toFloat() const;
        Vector<unsigned int> toUInt() const;

        enum Upconvert { UpconvertNone = 0x00  /* no conversion      */ };
#define OP_DECL(symbol, fun) \
        inline Vector<int> &operator symbol##=(const Vector<int> &x); \
        inline Vector<int> &operator symbol##=(const int &x); \
        inline Vector<int> operator symbol(const Vector<int> &x) const; \
        inline Vector<int> operator symbol(const int &x) const;

        OP_DECL(|, or_)
        OP_DECL(&, and_)
        OP_DECL(^, xor_)
#undef OP_DECL
    };

    template<typename Parent>
    struct VectorBase<unsigned int, Parent>
    {
        //operator unsigned int &() { return PARENT_DATA; }
        //operator unsigned int() const { return PARENT_DATA_CONST; }
        enum Upconvert { UpconvertNone = 0x00  /* no conversion      */ };
        Vector<float> toFloat() const;
        Vector<int> toInt() const;
    };
#undef PARENT_DATA
#undef PARENT_DATA_CONST

namespace VectorSpecialInitializerZero { enum Enum { Zero }; }
namespace VectorSpecialInitializerRandom { enum Enum { Random }; }

typedef bool Mask;
static const Mask kFullMask = true;
static inline Mask maskNthElement( int n ) { return 0 == n; }

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    protected:
        T m_data;
    public:
        typedef T Type;
        T &data() { return m_data; }
        T data() const { return m_data; }

        enum { Size = 1 };
        inline Vector() {}
        inline Vector(VectorSpecialInitializerZero::Enum) : m_data(0) {}
        inline Vector(VectorSpecialInitializerRandom::Enum) { makeRandom(); }
        inline Vector(const T &x) : m_data(x) {}
        template<typename Other> inline Vector(const Other *x) : m_data(x[0]) {}
        inline void makeZero() { m_data = 0; }
        inline void makeZero(Mask k) { if (k) m_data = 0; }
        inline void makeRandom() { m_data = std::rand(); }
        inline void makeRandom(Mask k) { if (k) m_data = std::rand(); }
        inline void store(void *mem) const { reinterpret_cast<T *>(mem)[0] = m_data; }
        inline void storeStreaming(void *mem) const { store(mem); }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return *this; }
        inline const Vector<T> badc() const { return *this; }
        inline const Vector<T> aaaa() const { return *this; }
        inline const Vector<T> bbbb() const { return *this; }
        inline const Vector<T> cccc() const { return *this; }
        inline const Vector<T> dddd() const { return *this; }
        inline const Vector<T> dbac() const { return *this; }

        Vector(const T *array, const Vector<int> &indexes) : m_data(array[indexes[0]]) {}
        void gather(const T *array, const Vector<int> &indexes) { m_data = array[indexes[0]]; }
        void gather(const T *array, const Vector<int> &indexes, Mask m) { if (m) m_data = array[indexes[0]]; }

        template<typename S> Vector(const S *array, const T S::* member, const Vector<int> &indexes, Mask = true, unsigned int arrayStride = sizeof(S))
            : m_data((&(array->*(member)))[arrayStride / sizeof(T) * indexes[0]]) {}

        template<typename S1, typename S2> Vector(const S1 *array, const S2 S1::* member1, const T S2::* member2, const Vector<int> &indexes, Mask = true, unsigned int arrayStride = sizeof(S1))
            : m_data((&(array->*(member1).*(member2)))[arrayStride / sizeof(T) * indexes[0]]) {}

        template<typename S> void gather(const S *array, const T S::* member, const Vector<int> &indexes, Mask = true, unsigned int arrayStride = sizeof(S))
        {
            m_data = (&(array->*(member)))[arrayStride / sizeof(T) * indexes[0]];
        }

        void scatter(T *array, const Vector<int> &indexes, Mask m ) const { if (m) array[indexes[0]] = m_data; }

        template<typename S> void scatter(S *array, const T S::* member, const Vector<int> &indexes, Mask m, unsigned int arrayStride = sizeof(S)) const
        {
            (&(array->*(member)))[arrayStride / sizeof(T) * indexes[0]] = m_data;
        }

        //prefix
        inline Vector &operator++() { return ++m_data; }
        //postfix
        inline Vector operator++(int) { return m_data++; }
        inline void increment(Mask mask) { if (mask) ++m_data; }
        inline void decrement(Mask mask) { if (mask) --m_data; }

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

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) { m_data symbol##= x.m_data; return *this; } \
        inline Vector &operator symbol##=(const T &x) { return operator symbol##=(Vector<T>(x)); } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(m_data symbol x.m_data); } \
        inline Vector operator symbol(const T &x) const { return operator symbol(Vector<T>(x)); }

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
        inline Mask operator symbol(const T &x) const { return operator symbol(Vector<T>(x)); }

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

        inline Vector &assign(const Vector<T> &v, const Mask &m) {
          if (m) m_data = v.m_data;
          return *this;
        }

        inline T max() const { return m_data; }
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator-(const T &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const T &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline Mask  operator< (const T &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline Mask  operator<=(const T &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline Mask  operator> (const T &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline Mask  operator>=(const T &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline Mask  operator==(const T &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline Mask  operator!=(const T &x, const Vector<T> &v) { return Vector<T>(x) != v; }

#define PARENT_DATA (static_cast<Vector<int> *>(this)->m_data)
#define PARENT_DATA_CONST (static_cast<const Vector<int> *>(this)->m_data)
#define OP_IMPL(symbol, fun) \
  template<> inline Vector<int> &VectorBase<int, Vector<int> >::operator symbol##=(const Vector<int> &x) { PARENT_DATA symbol##= x.m_data; return *static_cast<Vector<int> *>(this); } \
  template<> inline Vector<int> &VectorBase<int, Vector<int> >::operator symbol##=(const int &x) { return operator symbol##=(Vector<int>(x)); } \
  template<> inline Vector<int> VectorBase<int, Vector<int> >::operator symbol(const Vector<int> &x) const { return Vector<int>(PARENT_DATA_CONST symbol x.m_data); } \
  template<> inline Vector<int> VectorBase<int, Vector<int> >::operator symbol(const int &x) const { return operator symbol(Vector<int>(x)); }
  OP_IMPL(&, and_)
  OP_IMPL(|, or_)
  OP_IMPL(^, xor_)
#undef OP_IMPL
#undef ALIGN

  template<> inline Vector<float> VectorBase<int, Vector<int> >::toFloat() const { return Vector<float>(static_cast<float>(static_cast<const Vector<int> *>(this)->m_data)); }
  template<> inline Vector<float> VectorBase<unsigned int, Vector<unsigned int> >::toFloat() const { return Vector<float>(static_cast<float>(static_cast<const Vector<unsigned int> *>(this)->m_data)); }
  template<> inline Vector<int> VectorBase<float, Vector<float> >::toInt() const { return Vector<int>(static_cast<int>(static_cast<const Vector<float> *>(this)->m_data)); }
  template<> inline Vector<unsigned int> VectorBase<float, Vector<float> >::toUInt() const { return Vector<unsigned int>(static_cast<unsigned int>(static_cast<const Vector<float> *>(this)->m_data)); }
  template<> inline Vector<int> VectorBase<unsigned int, Vector<unsigned int> >::toInt() const { return Vector<int>(static_cast<int>(static_cast<const Vector<unsigned int> *>(this)->m_data)); }
  template<> inline Vector<unsigned int> VectorBase<int, Vector<int> >::toUInt() const { return Vector<unsigned int>(static_cast<unsigned int>(static_cast<const Vector<int> *>(this)->m_data)); }
#undef PARENT_DATA_CONST
#undef PARENT_DATA

  template<typename T> static inline Simple::Vector<T> min (const Simple::Vector<T> &x, const T &y) { return std::min( x.data(), y ); }
  template<typename T> static inline Simple::Vector<T> max (const Simple::Vector<T> &x, const T &y) { return std::max( x.data(), y ); }
  template<typename T> static inline Simple::Vector<T> min (const T &x, const Simple::Vector<T> &y) { return std::min( x, y.data() ); }
  template<typename T> static inline Simple::Vector<T> max (const T &x, const Simple::Vector<T> &y) { return std::max( x, y.data() ); }
  template<typename T> static inline Simple::Vector<T> min (const Simple::Vector<T> &x, const Simple::Vector<T> &y) { return std::min( x.data(), y.data() ); }
  template<typename T> static inline Simple::Vector<T> max (const Simple::Vector<T> &x, const Simple::Vector<T> &y) { return std::max( x.data(), y.data() ); }
  template<typename T> static inline Simple::Vector<T> sqrt(const Simple::Vector<T> &x) { return std::sqrt( x.data() ); }
  template<typename T> static inline Simple::Vector<T> abs (const Simple::Vector<T> &x) { return std::abs( x.data() ); }
  template<typename T> static inline Simple::Vector<T> sin (const Simple::Vector<T> &x) { return std::sin( x.data() ); }
  template<typename T> static inline Simple::Vector<T> cos (const Simple::Vector<T> &x) { return std::cos( x.data() ); }
} // namespace Simple

//X #include <iostream>
//X 
//X template<typename T>
//X static inline std::ostream &operator<<( std::ostream &out, const Simple::Vector<T> &v )
//X {
//X   return out << v[0];
//X }
//X 
//X template<typename T>
//X static inline std::istream &operator>>( std::istream &in, Simple::Vector<T> &v )
//X {
//X   return in >> reinterpret_cast<T &>( v );
//X }

#endif // SIMPLE_VECTOR_H
