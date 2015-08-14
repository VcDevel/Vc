#include "../tests/unittest.h"

TEST(test_sqrt) {
    int test = 1 + 1;
    COMPARE(test, 3) << "more details";
    VERIFY(1 < 0);
}

