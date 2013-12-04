/*{{{
    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#include <Vc/Vc>

#include <array>
#include <algorithm>

#include "../tsc.h"

using Vc::double_v;
using Vc::double_m;

constexpr size_t ArraySize = 10240;

using Point = std::array<double_v, 3>;
using PointArray = std::array<Point, ArraySize / double_v::Size>;

static const std::array<double_v, 3> origin  = {{ 0.2, 0.3, 0.4 }};
static const std::array<double_v, 3> boxsize = {{ 0.5, 0.3, 0.1 }};

double_m contains(const Point &point)
{
    double_m inside[3];
    for (int dir = 0; dir < 3; ++dir) {
        inside[dir] = abs(point[dir] - origin[dir]) < boxsize[dir];
    }
    return inside[0] && inside[1] && inside[2];
}

std::array<bool, ArraySize> contains(const PointArray &points)
{
    std::array<bool, ArraySize> inside;
    auto storeIt = inside.begin();
    for (const auto &p : points) {
        contains(p).store(storeIt);
        storeIt += double_v::Size;
    }
    return inside;
}

std::array<bool, ArraySize> g_inside;

int main()
{
    PointArray points;
    std::generate(points.begin(), points.end(), []() -> Point {
        return {{ double_v::Random(), double_v::Random(), double_v::Random() }};
    });

    TimeStampCounter tsc;
    tsc.start();
    const auto &tmp = contains(points);
    tsc.stop();
    g_inside = tmp;
    std::cout << tsc.cycles() << std::endl;

    return 0;
}
