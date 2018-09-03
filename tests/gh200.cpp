#include "unittest.h"
#include <Vc/Vc>

TEST_TYPES(V, rounding, RealVectors)
{
    using T = typename V::value_type;
    for (T d : { 1e100, 2.1, 2.9, 1.0, 2.0, 3.0, -2.1, -2.9, -1e100 }) {
        V v(d);
        COMPARE(Vc::trunc(v), V(std::trunc(d))) << "d = " << d;
        COMPARE(Vc::floor(v), V(std::floor(d))) << "d = " << d;
        COMPARE(Vc::ceil(v), V(std::ceil(d))) << "d = " << d;
    }
}
