#include <Vc/Vc>

#define CAT(a, b) a##b
#define name(a, b) CAT(a, b)

using namespace Vc;
float_v
#ifdef Vc_MSVC
__declspec(dllexport)
#endif
name(fooLib1, POSTFIX)(const float_v &a)
{
    const float_v b = min(a, float_v(1));
    std::cerr << b;
    return b;
}
