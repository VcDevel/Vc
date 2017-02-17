#include <Vc/Vc>

using namespace Vc;
float_v foo0(const float_v &a)
{
    const float_v b = max(a, float_v(1));
    const Vc::datapar<float> c = 1;
    std::cerr << b << c;
    return b;
}
