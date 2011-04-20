set(CTEST_PROJECT_NAME "Vc")
set(CTEST_NIGHTLY_START_TIME "00:00:00 CET")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "code.compeng.uni-frankfurt.de")
set(CTEST_DROP_LOCATION "/dashboard/submit.php?project=Vc")

set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_PROJECT_SUBPROJECTS
   other
   Scalar
   SSE
   LRBni
   AVX
   )

set(CTEST_UPDATE_TYPE "git")

find_program(GITCOMMAND git)
set(CTEST_UPDATE_COMMAND "${GITCOMMAND}")

mark_as_advanced(GITCOMMAND)
