#include <Vc/simd>

void test_fixed_simd(void *mem)
{
    new(mem) Vc::fixed_size_simd<float, Vc::simd_abi::max_fixed_size<float>>;
}

void test_fixed_mask(void *mem)
{
    new(mem) Vc::fixed_size_simd_mask<float, Vc::simd_abi::max_fixed_size<float>>;
}

void test_native_simd(void *mem)
{
    new(mem) Vc::native_simd<float>;
}

void test_native_mask(void *mem)
{
    new(mem) Vc::native_simd_mask<float>;
}

int main() { return 0; }
