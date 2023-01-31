@REM Copyright 2022 prime-slam
@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at
@REM     http://www.apache.org/licenses/LICENSE-2.0
@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.

SETLOCAL EnableDelayedExpansion

SET plat="x64" || EXIT /B !ERRORLEVEL!

:: Early check for build tools
cmake --version || EXIT /B !ERRORLEVEL!

pushd %~dp0
cd ..

for /D %%P in (C:\hostedtoolcache\windows\Python\3*) do CALL :build %plat% %%P\%~1\python.exe || popd && EXIT /B !ERRORLEVEL!

popd
EXIT /B 0

:build
cmake -S . -B build -G "Visual Studio 16 2019" -A "%~1" -DBUILD_PYTHON=ON -DBUILD_TESTS=OFF -DPYTHON_EXECUTABLE:FILEPATH=%~2  || EXIT /B !ERRORLEVEL!
cmake --build build --config Release --target build-wheel || EXIT /B !ERRORLEVEL!
EXIT /B 0