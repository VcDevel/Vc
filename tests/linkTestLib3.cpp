#include <Vc/Vc>
#include <Vc/IO>

using namespace Vc;
float_v fooLib3(float_v::AsArg a)
{
    const float_v b = a + float_v::One();
    std::cerr << b;
    return b;
}
