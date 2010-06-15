SET(CTEST_DROP_METHOD "http")
SET(CTEST_DROP_SITE "fairroot.gsi.de")
SET(CTEST_DROP_LOCATION "/CDash/submit.php?project=VC")
SET(CTEST_UPDATE_TYPE git)
SET(CTEST_TRIGGER_SITE "fairroot.gsi.de")

find_program(GITCOMMAND git)
mark_as_advanced(GITCOMMAND)

SET(UPDATE_COMMAND "${GITCOMMAND}")
SET(UPDATE_OPTIONS pull)
