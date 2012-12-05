#include <Vc/Vc>
#include <Vc/IO>

using namespace Vc;
float_v
#ifdef VC_MSVC
__declspec(dllexport)
#endif
fooLib1(float_v::AsArg a)
{
    const float_v b = a + float_v::One();
    std::cerr << b;
    return b;
}
