/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef RUNTIMEMEAN_H
#define RUNTIMEMEAN_H

#include "../tsc.h"
#include <iostream>

class RuntimeMean
{
    TimeStampCounter tsc;
    unsigned long cycles;
    int count;
public:
    RuntimeMean()
        : cycles(0), count(0)
    {}
    inline void start() { tsc.start(); }
    inline void stop() {
        tsc.stop();
        cycles += tsc.cycles();
        ++count;
    }
    ~RuntimeMean()
    {
        std::cout << "runtime mean: " << cycles / count << '\n';
    }
};

#endif // RUNTIMEMEAN_H
