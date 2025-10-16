"""
Build script for the pybind11 CUDA extension
"""
import os
import subprocess
import sys
import platform
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create build directory
    build_dir = script_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure with cmake
    print("Configuring with CMake...")
    cmake_args = [
        "cmake", "..", 
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPYTHON_EXECUTABLE=" + sys.executable
    ]
    
    # Add CUDA architecture if not set
    if not os.environ.get('CMAKE_CUDA_ARCHITECTURES'):
        cmake_args.extend(["-DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86"])
    
    result = subprocess.run(cmake_args, cwd=build_dir)
    
    if result.returncode != 0:
        print("CMake configuration failed!")
        return 1
    
    # Build
    print("Building...")
    if platform.system() == "Windows":
        # Use MSBuild on Windows
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
    else:
        # Use make on Unix-like systems
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
    
    result = subprocess.run(build_cmd, cwd=build_dir)
    
    if result.returncode != 0:
        print("Build failed!")
        return 1
    
    print("Build successful!")
    print(f"Extension should be in: {build_dir}")
    
    # List the built files
    print("\nBuilt files:")
    for file in build_dir.rglob("*.pyd"):  # Windows
        print(f"  {file}")
    for file in build_dir.rglob("*.so"):   # Linux/Mac
        print(f"  {file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
