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
#include <limits>

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

template<> class PseudoRandom<Vc::float_v>
{
    public:
        static Vc::float_v next();

    private:
        static Vc::uint_v state;
};

Vc::uint_v PseudoRandom<Vc::float_v>::state(((Vc::uint_v(Vc::IndexesFromZero) + std::rand()) * 1103515245 + 12345) * 1103515245 + 12345);

inline Vc::float_v PseudoRandom<Vc::float_v>::next()
{
    using Vc::float_v;
    static const float_v mask(2.f - std::numeric_limits<float>::epsilon()); // 0x3fffffff
    const float_v ret = ((state.reinterpretCast<float_v>() & mask) | float_v::One()) - float_v::One(); // [1.0, 2.0) - 1.0 => [0.0, 1.0)
    state = (state * 1103515245 + 12345);
    return ret;
}

template<> class PseudoRandom<Vc::double_v>
{
    public:
        static Vc::double_v next();

    private:
        static Vc::uint_v state;
};

Vc::uint_v PseudoRandom<Vc::double_v>::state(((Vc::uint_v(Vc::IndexesFromZero) + std::rand()) * 1103515245 + 12345) * 1103515245 + 12345);

inline Vc::double_v PseudoRandom<Vc::double_v>::next()
{
    using Vc::double_v;
    using Vc::uint_v;
    static const double_v mask(2. - std::numeric_limits<double>::epsilon()); // 0x3fffffffffffffff
    if (sizeof(double_v) == sizeof(state)) {
        const double_v ret = ((state.reinterpretCast<double_v>() & mask) | double_v::One()) - double_v::One(); // [1.0, 2.0) - 1.0 => [0.0, 1.0)
        state = (state * 1103515245 + 12345);
        return ret;
    } else if (double_v::Size == 1 && uint_v::Size == 1) {
        union {
            double_v::EntryType d;
            uint_v::EntryType i[2];
        } x;
        x.i[0] = state[0];
        state = (state * 1103515245 + 12345);
        x.i[1] = state[0];
        state = (state * 1103515245 + 12345);
        return ((double_v(x.d) & mask) | double_v::One()) - double_v::One(); // [1.0, 2.0) - 1.0 => [0.0, 1.0)
    }
}

#ifdef VC_IMPL_SSE
template<> class PseudoRandom<Vc::sfloat_v>
{
    public:
        static Vc::sfloat_v next();
};

inline Vc::sfloat_v PseudoRandom<Vc::sfloat_v>::next()
{
    return Vc::SSE::M256::create(PseudoRandom<Vc::float_v>::next().data(),
            PseudoRandom<Vc::float_v>::next().data());
}
#endif

#endif // RANDOM_H
