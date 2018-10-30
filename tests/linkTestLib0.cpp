#include <Vc/simd>

#define CAT(a, b) a##b
#define name(a, b) CAT(a, b)

using namespace Vc;
native_simd<float> name(fooLib0, POSTFIX)(const native_simd<float> &a)
{
    const native_simd<float> b = max(a, native_simd<float>(1));
    return b;
}
