#include <Vc/Vc>

Vc::float_v test(Vc::float_v a, Vc::float_v b)
{
    return a + b;
}

Vc::float_m mask_test(Vc::float_m a, Vc::float_m b)
{
    return a & b;
}

int Vc_CDECL main() { return 0; }
