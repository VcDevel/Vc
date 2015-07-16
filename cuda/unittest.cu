#include "../tests/unittest.h"

TEST(test_sqrt) {
    float test = sqrtf(4.0f);
    COMPARE(test, 2.0f) << "not working!";
    VERIFY(1 > 0);
}

