@SET SUBDIR=%~dp0
@cd %SUBDIR%

@ctest -V -S test.cmake
@SET PATH=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin;%PATH%
@ctest -V -S test.cmake
