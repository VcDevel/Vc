#include <Vc/datapar>

using float_v = Vc::datapar<float, Vc::datapar_abi::native<float>>;
using float_m = typename float_v::mask_type;

float_v test(float_v a, float_v b)
{
    return a + b;
}

float_m mask_test(float_m a, float_m b)
{
    return a & b;
}

int Vc_CDECL main() { return 0; }
