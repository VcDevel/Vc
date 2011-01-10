/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_CPUSET_H
#define VC_CPUSET_H

#include <sched.h>

#ifdef __cplusplus
extern "C" {
#endif

int cpuIsSet(size_t cpucount, const cpu_set_t *cpumask);
int cpuCount(const cpu_set_t *cpumask);
void cpuZero(cpu_set_t *mask);
void cpuSet(size_t id, cpu_set_t *mask);

#ifdef __cplusplus
}
#endif

#endif
