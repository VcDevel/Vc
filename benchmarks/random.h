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

#ifndef RANDOM_H
#define RANDOM_H

#include <Vc/Vc>
#include <cstdlib>

// this is not a random number generator
template<typename Vector> class PseudoRandom
{
    public:
        static Vector next();

    private:
        static Vector state;
};

template<> Vc::uint_v PseudoRandom<Vc::uint_v>::state(Vc::uint_v(Vc::IndexesFromZero) + std::rand());
template<> Vc::int_v PseudoRandom<Vc::int_v>::state(Vc::int_v(Vc::IndexesFromZero) + std::rand());

template<> inline Vc::uint_v PseudoRandom<Vc::uint_v>::next()
{
    state = (state * 1103515245 + 12345);
    return (state >> 16) | (state << 16); // rotate
}

template<> inline Vc::int_v PseudoRandom<Vc::int_v>::next()
{
    state = (state * 1103515245 + 12345);
    return (state >> 16) | (state << 16); // rotate
}

#if !VC_IMPL_LRBni
template<> Vc::ushort_v PseudoRandom<Vc::ushort_v>::state(Vc::ushort_v(Vc::IndexesFromZero) + std::rand());
template<> Vc::short_v PseudoRandom<Vc::short_v>::state(Vc::short_v(Vc::IndexesFromZero) + std::rand());

template<> inline Vc::ushort_v PseudoRandom<Vc::ushort_v>::next()
{
    state = (state * 257 + 24151);
    return (state >> 8) | (state << 8); // rotate
}

template<> inline Vc::short_v PseudoRandom<Vc::short_v>::next()
{
    state = (state * 257 + 24151);
    return (state >> 8) | (state << 8); // rotate
}
#endif

template<> class PseudoRandom<Vc::float_v>
{
    public:
        static Vc::float_v next();

    private:
        static Vc::uint_v state;
};

Vc::uint_v PseudoRandom<Vc::float_v>::state(Vc::uint_v(Vc::IndexesFromZero) + std::rand());

inline Vc::float_v PseudoRandom<Vc::float_v>::next()
{
    const Vc::float_v ConvertFactor = 1.f / (~0u);
    state = (state * 1103515245 + 12345);
    return Vc::float_v(state) * ConvertFactor;
}

template<> class PseudoRandom<Vc::double_v>
{
    public:
        static Vc::double_v next();

    private:
        static Vc::uint_v state;
};

Vc::uint_v PseudoRandom<Vc::double_v>::state(Vc::uint_v(Vc::IndexesFromZero) + std::rand());

inline Vc::double_v PseudoRandom<Vc::double_v>::next()
{
    const Vc::double_v ConvertFactor = 1. / (~0u);
    state = (state * 1103515245 + 12345);
    return Vc::double_v(state) * ConvertFactor;
}

#if VC_IMPL_SSE
template<> class PseudoRandom<Vc::sfloat_v>
{
    public:
        static Vc::sfloat_v next();

    private:
        static Vc::ushort_v state;
};

Vc::ushort_v PseudoRandom<Vc::sfloat_v>::state(Vc::ushort_v(Vc::IndexesFromZero) + std::rand());

inline Vc::sfloat_v PseudoRandom<Vc::sfloat_v>::next()
{
    const Vc::sfloat_v ConvertFactor = 1.f / (~0u);
    state = (state * 257 + 24151);
    return Vc::sfloat_v(state) * ConvertFactor;
}
#endif

#endif // RANDOM_H
