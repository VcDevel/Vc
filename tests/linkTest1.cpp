#include <Vc/Vc>
#include <Vc/IO>
#include <common/support.h>

using namespace Vc;

float_v fooLib0(float_v::AsArg a);
float_v fooLib1(float_v::AsArg a);
float_v fooLib2(float_v::AsArg a);
float_v fooLib3(float_v::AsArg a);
float_v foo0(float_v::AsArg a);
float_v foo1(float_v::AsArg a)
{
    const float_v b = a + float_v::One();
    std::cerr << b;
    return b;
}

int main()
{
    float_v x = float_v::Random();
    x = fooLib0(fooLib1(fooLib2(fooLib3(foo0(foo1(x))))));
    return static_cast<int>(x.sum());
}
