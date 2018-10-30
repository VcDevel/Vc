#include <experimental/simd>

using float_v = std::experimental::simd<float, std::experimental::simd_abi::_GLIBCXX_SIMD_ABI>;
using float_m = typename float_v::mask_type;

float_v test(float_v a, float_v b)
{
    return a + b;
}

float_m mask_test(float_m a, float_m b)
{
    return a & b;
}

int main() { return 0; }
