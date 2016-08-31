#include <Vc/Vc>
#include <Vc/IO>
#include <Vc/datapar>
#include <Vc/support.h>

using namespace Vc;
float_v foo0(float_v::AsArg a)
{
    const float_v b = sin(a + float_v::One());
    const Vc::datapar<float> c = 1;
    std::cerr << b << c;
    return b;
}
