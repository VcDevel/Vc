#include <Vc/datapar>

void test_fixed_datapar(void *mem)
{
    new(mem) Vc::fixed_size_datapar<float, Vc::datapar_abi::max_fixed_size>;
}

void test_fixed_mask(void *mem)
{
    new(mem) Vc::fixed_size_mask<float, Vc::datapar_abi::max_fixed_size>;
}

void test_native_datapar(void *mem)
{
    new(mem) Vc::native_datapar<float>;
}

void test_native_mask(void *mem)
{
    new(mem) Vc::native_mask<float>;
}

int main() { return 0; }
