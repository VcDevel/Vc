#include <experimental/simd>

void test_fixed_simd(void *mem)
{
    new(mem) std::experimental::fixed_size_simd<float, std::experimental::simd_abi::max_fixed_size<float>>;
}

void test_fixed_mask(void *mem)
{
    new(mem) std::experimental::fixed_size_simd_mask<float, std::experimental::simd_abi::max_fixed_size<float>>;
}

void test_native_simd(void *mem)
{
    new(mem) std::experimental::native_simd<float>;
}

void test_native_mask(void *mem)
{
    new(mem) std::experimental::native_simd_mask<float>;
}

int main() { return 0; }
