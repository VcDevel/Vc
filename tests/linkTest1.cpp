#include <Vc/Vc>

using namespace Vc;

float_v fooLib0A(const float_v &a);
float_v fooLib1A(const float_v &a);
float_v fooLib0B(const float_v &a);
float_v fooLib1B(const float_v &a);
float_v fooLib2(const float_v &a);
float_v fooLib3(const float_v &a);
float_v foo0(const float_v &a);
float_v foo1(const float_v &a)
{
    const float_v b = max(a, float_v(1));
    const Vc::datapar<float> c = 1;
    std::cerr << b << c;
    return b;
}

int Vc_CDECL main()
{
    float_v x{[](unsigned i) -> float { return (i + 0x7fffffffu) * 298592097u; }};
    x = fooLib0A(fooLib0B(fooLib1A(fooLib1B(fooLib2(fooLib3(foo0(foo1(x))))))));
    return static_cast<int>(reduce(x)) >> 8;
}
