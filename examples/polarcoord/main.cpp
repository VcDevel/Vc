/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

*/

#include <Vc/Vc>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using Vc::float_v;

typedef Vc::Memory<float_v, 1000> Mem10;

static const float_v TwoOverRandMax(2.f / RAND_MAX);

int main()
{
    // allocate memory for our initial x and y coordinates. Note that you can also put it into a
    // normal float C-array but that you then must ensure alignment to Vc::VectorAlignment!
    Mem10 x_mem;
    Mem10 y_mem;
    Mem10 r_mem;
    Mem10 phi_mem;

    // fill the memory with values from -1.f to 1.f (a proper implementation would use a vectorized
    // RNG).
    for (unsigned int i = 0; i < x_mem.vectorsCount(); ++i) {
        Vc::Memory<float_v, 2 * float_v::Size> m;
        // the following makes sure we get the same random number sequences regardless of float_v::Size
        for (unsigned int j = 0; j < float_v::Size; ++j) {
            m[j] = std::rand();
            m[j + float_v::Size] = std::rand();
        }
        x_mem.vector(i) = m.vector(0) * TwoOverRandMax - 1.f;
        y_mem.vector(i) = m.vector(1) * TwoOverRandMax - 1.f;
    }

    // calculate the polar coordinates for all coordinates and overwrite the euclidian coordinates
    // with the result
    for (unsigned int i = 0; i < x_mem.vectorsCount(); ++i) {
        const float_v x = x_mem.vector(i);
        const float_v y = y_mem.vector(i);

        r_mem.vector(i) = Vc::sqrt(x * x + y * y);
        float_v phi = Vc::atan2(y, x) * (180. / M_PI);
        phi(phi < 0.f) += 360.f;
        phi_mem.vector(i) = phi;
    }

    // print the results
    for (unsigned int i = 0; i < x_mem.entriesCount(); ++i) {
        std::cout << std::setw(3) << i << ": ";
        std::cout << std::setw(10) << x_mem[i] << ", " << std::setw(10) << y_mem[i] << " -> ";
        std::cout << std::setw(10) << r_mem[i] << ", " << std::setw(10) << phi_mem[i] << '\n';
    }

    return 0;
}
