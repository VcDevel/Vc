#include <Vc/Vc>

Vc::float_v test_vector_abi(Vc::float_v);
Vc::float_m test_mask_abi(Vc::float_m);

Vc::float_v a_v, b_v;
Vc::float_m a_m, b_m;

void my_test1()
{
    asm volatile("vector_abi_start:");
    b_v = test_vector_abi(a_v);
    asm volatile("vector_abi_end:");
}

void my_test2()
{
    asm volatile("mask_abi_start:");
    b_m = test_mask_abi(a_m);
    asm volatile("mask_abi_end:");
}
