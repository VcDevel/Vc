#include <Vc/simd>

using namespace Vc;
native_simd<float> foo0(const native_simd<float> &a)
{
    const native_simd<float> b = max(a, native_simd<float>(1));
    return b;
}
