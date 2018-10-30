#include <Vc/simd>

using namespace Vc;

native_simd<float> fooLib0A(const native_simd<float> &a);
native_simd<float> fooLib1A(const native_simd<float> &a);
native_simd<float> fooLib0B(const native_simd<float> &a);
native_simd<float> fooLib1B(const native_simd<float> &a);
native_simd<float> fooLib2(const native_simd<float> &a);
native_simd<float> fooLib3(const native_simd<float> &a);
native_simd<float> foo0(const native_simd<float> &a);
native_simd<float> foo1(const native_simd<float> &a)
{
    const native_simd<float> b = max(a, native_simd<float>(1));
    return b;
}

int main()
{
    native_simd<float> x{[](unsigned i) -> float { return (i + 0x7fffffffu) * 298592097u; }};
    x = fooLib0A(fooLib0B(fooLib1A(fooLib1B(fooLib2(fooLib3(foo0(foo1(x))))))));
    return static_cast<int>(reduce(x)) >> 8;
}
