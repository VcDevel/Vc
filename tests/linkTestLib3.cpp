#include <Vc/Vc>

using namespace Vc;
float_v fooLib3(const float_v &a)
{
    const float_v b = min(a, float_v(1));
    std::cerr << b;
    return b;
}
