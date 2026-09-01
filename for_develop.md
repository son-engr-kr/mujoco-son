# prepare build
```bash
winget install Ninja-build.Ninja

winget install -e --id Microsoft.VisualStudio.2022.BuildTools
```

# build
in the `x64 Native Tools Command Prompt for VS 2022`

or

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

## Clean build (recommended when CMake configuration changes)
```bash
if exist build\CMakeCache.txt del /F /Q build\CMakeCache.txt && cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build -j %NUMBER_OF_PROCESSORS%
```

## Regular build
```bash
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
ninja -C build -j %NUMBER_OF_PROCESSORS%
```

# Deploy(After build)
```bash
.venv\Scripts\python.exe -m pip install -r python\build_requirements.txt
```
git bash-create sdist
```bash
cd "$PROJECT_DIR"
export PROJECT_DIR=$(pwd)
export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/Scripts:$PATH"

cd "$PROJECT_DIR/python"
bash make_sdist.sh
```

```bash
cd "$PROJECT_DIR"
export MUJOCO_PATH=$(cygpath -w "$PROJECT_DIR")
export MUJOCO_PLUGIN_PATH=$(cygpath -w "$PROJECT_DIR/build/bin")

pip wheel --no-deps --no-build-isolation \
  "$PROJECT_DIR/python/dist/mujoco-3.3.3+son4.0a2.tar.gz" \
  -w "$PROJECT_DIR/python/dist"
```
- make a new tag
- upload tar.gz, whl files in dist folder
- copy link of whl file and use it when pip install

# run simulate
```bash
build\bin\simulate.exe
```



# Test

## Teleport muscle test (forward test)
```
cmake --build build --target teleport_muscle
```
```
build\bin\teleport_muscle model_compliant_muscle_test.xml 1
```