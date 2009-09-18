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

enum {
    Count = 64,
    OuterCount = Count / float_v::Size
};

static const float TwoOverRandMax = 2.f / RAND_MAX;

int main()
{
    // allocate memory for our initial x and y coordinates. Note that you can also put it into a
    // normal float C-array but that you then must ensure alignment to Vc::VectorAlignment!
    float_v::Memory x_mem[OuterCount];
    float_v::Memory y_mem[OuterCount];

    // fill the memory with values from -1.f to 1.f (a proper implementation would use a vectorized
    // RNG).
    for (int i = 0; i < OuterCount; ++i) {
        for (int j = 0; j < float_v::Size; ++j) {
            x_mem[i][j] = std::rand() * TwoOverRandMax - 1.f;
            y_mem[i][j] = std::rand() * TwoOverRandMax - 1.f;

            std::cout << std::setw(3) << i * float_v::Size + j << ": ";
            std::cout << std::setw(10) << x_mem[i][j] << ", " << std::setw(10) << y_mem[i][j] << '\n';
        }
    }
    std::cout << "---------------------------\n";

    // calculate the polar coordinates for all coordinates and overwrite the euclidian coordinates
    // with the result
    for (int i = 0; i < OuterCount; ++i) {
        float_v x(x_mem[i]);
        float_v y(y_mem[i]);

        float_v r = Vc::sqrt(x * x + y * y);
        float_v phi = Vc::atan2(y, x) * (180. / M_PI);
        phi(phi < 0.f) += 360.f;

        r.store(x_mem[i]);
        phi.store(y_mem[i]);
    }


    // print the results
    for (int i = 0; i < OuterCount; ++i) {
        for (int j = 0; j < float_v::Size; ++j) {
            std::cout << std::setw(3) << i * float_v::Size + j << ": ";
            std::cout << std::setw(10) << x_mem[i][j] << ", " << std::setw(10) << y_mem[i][j] << '\n';
        }
    }

    std::cout << std::flush;

    return 0;
}
