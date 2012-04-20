get_filename_component(_currentDir "${CMAKE_CURRENT_LIST_FILE}" PATH)
include("${_currentDir}/cmake/OptimizeForArchitecture.cmake")

set(CMAKE_SYSTEM_NAME "Linux")
AutodetectHostArchitecture()
message("${TARGET_ARCHITECTURE}")
