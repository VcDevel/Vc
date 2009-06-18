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

#ifndef VECIO_H
#define VECIO_H

#include "vector.h"
#include <iostream>

#ifdef ENABLE_LARRABEE
#define VECTOR_NAMESPACE Larrabee
#elif defined(USE_SSE)
#define VECTOR_NAMESPACE SSE
#else
#define VECTOR_NAMESPACE Simple
#endif

namespace
{
    namespace AnsiColor
    {
        static const char *const green  = "\033[1;40;32m";
        static const char *const yellow = "\033[1;40;33m";
        static const char *const blue   = "\033[1;40;34m";
        static const char *const normal = "\033[0m";
    } // namespace AnsiColor
} // anonymous namespace

template<typename T>
std::ostream &operator<<(std::ostream &out, const VECTOR_NAMESPACE::Vector<T> &v)
{
    out << AnsiColor::green << "[";
    for (int i = 0; i < v.Size; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << v[i];
    }
    out << "]" << AnsiColor::normal;
    return out;
}

#ifdef VC_HAVE_FMA
template<typename T>
std::ostream &operator<<(std::ostream &out, const VECTOR_NAMESPACE::VectorMultiplication<T> &v)
{
    return out << VECTOR_NAMESPACE::Vector<T>(v);
}
#endif

template<typename T>
std::ostream &operator<<(std::ostream &out, const typename VECTOR_NAMESPACE::Vector<T>::Mask &m)
{
    out << AnsiColor::yellow << "m[";
    for (unsigned int i = 0; i < VECTOR_NAMESPACE::Vector<T>::Size; ++i) {
        if (i > 0 && (i % 4) == 0) {
            out << " ";
        }
        out << m[i];
    }
    out << "]" << AnsiColor::normal;
    return out;
}

#undef VECTOR_NAMESPACE

#endif // VECIO_H
