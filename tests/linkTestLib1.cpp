#include <Vc/Vc>
#include <iostream>

#define CAT(a, b) a##b
#define name(a, b) CAT(a, b)

using namespace Vc;
float_v name(fooLib1, POSTFIX)(const float_v &a)
{
    const float_v b = min(a, float_v(1));
    std::cerr << b;
    return b;
}
