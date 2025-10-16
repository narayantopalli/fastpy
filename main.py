import numpy as np
import sys
import os
import time

# Add the build directory to the Python path
build_dir = os.path.join(os.path.dirname(__file__), "build/Release")
sys.path.insert(0, build_dir)

def benchmark(fastpy_module, A, B, num_runs=5):
    """Quick benchmark with warmup"""

    A_d = fastpy_module.fastArray(A, fastpy_module.CUDA)
    B_d = fastpy_module.fastArray(B, fastpy_module.CUDA)

    # Warmup
    for _ in range(5):
        _ = fastpy_module.matmul(A_d, B_d)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        C_d = fastpy_module.matmul(A_d, B_d)
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    M, N, K = A.shape[A.ndim-2], B.shape[B.ndim-1], A.shape[A.ndim-1]
    flops = 2 * M * N * K
    
    # Calculate total operations considering broadcasting
    # For broadcasting, we need to consider the output shape
    if A.ndim > 2 or B.ndim > 2:
        # Get the output shape from numpy broadcasting
        output_shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        total_ops = 1
        for dim in output_shape:
            total_ops *= dim
        flops *= total_ops
    
    # Handle very fast operations
    if avg_time < 1e-6:  # Less than 1 microsecond
        gflops = flops / 1e-6  # Use 1 microsecond as minimum
        avg_time = 1e-6
    else:
        gflops = flops / (avg_time * 1e9)
    
    return avg_time, gflops

def test_correctness(fastpy_module, A, B):
    """Test correctness against numpy"""
    A_d = fastpy_module.fastArray(A, fastpy_module.CUDA)
    B_d = fastpy_module.fastArray(B, fastpy_module.CUDA)
    C_d = fastpy_module.matmul(A_d, B_d)
    C_numpy = np.matmul(A, B)
    return np.allclose(C_d.to_numpy(), C_numpy, rtol=1e-3, atol=1e-3)

def test_4d_tensors(fastpy_module):
    """Test 4D tensor operations"""
    print("\n" + "="*50)
    print("4D TENSOR TESTS")
    print("="*50)
    
    test_cases = [
        {
            "name": "4D x 4D (Batch x Batch x M x K)",
            "A_shape": (2, 3, 64, 128),
            "B_shape": (2, 3, 128, 256),
            "expected": (2, 3, 64, 256)
        },
        {
            "name": "2D x 4D Broadcasting",
            "A_shape": (64, 128),
            "B_shape": (2, 3, 128, 256),
            "expected": (2, 3, 64, 256)
        },
        {
            "name": "4D x 2D Broadcasting", 
            "A_shape": (2, 3, 64, 128),
            "B_shape": (128, 256),
            "expected": (2, 3, 64, 256)
        },
        {
            "name": "Large 4D Test",
            "A_shape": (1024, 8, 64, 48),
            "B_shape": (1024, 8, 48, 64),
            "expected": (1024, 8, 64, 64)
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"  A: {test['A_shape']}, B: {test['B_shape']}")
        
        A = np.random.randn(*test['A_shape']).astype(np.float32)
        B = np.random.randn(*test['B_shape']).astype(np.float32)
        
        # Test regular matmul correctness
        if test_correctness(fastpy_module, A, B):
            print("  ✓ Regular matmul: PASS")
            
            # Benchmark
            avg_time, gflops = benchmark(fastpy_module, A, B)
            print(f"  ⚡ Regular: {avg_time*1000:.2f}ms, {gflops:.1f} GFLOPS")
        else:
            print("  ✗ Regular matmul: FAIL")

def test_standard_matmul(fastpy_module):
    """Test standard matrix multiplication"""
    print("\n" + "="*50)
    print("STANDARD MATRIX MULTIPLICATION")
    print("="*50)
    
    sizes = [
        {"name": "Tiny", "M": 16, "N": 16, "K": 16, "D": 8},
        {"name": "Small", "M": 256, "N": 256, "K": 256, "D": 16},
        {"name": "Medium", "M": 512, "N": 512, "K": 512, "D": 32},
        {"name": "Large", "M": 1024, "N": 1024, "K": 1024, "D": 64}
    ]
    
    for size in sizes:
        print(f"\n{size['name']} ({size['M']}x{size['N']}x{size['K']})")
        
        A = np.random.randn(size['D'], size['M'], size['K']).astype(np.float32)
        B = np.random.randn(size['D'], size['K'], size['N']).astype(np.float32)
        
        # Test regular matmul
        if test_correctness(fastpy_module, A, B):
            print("  ✓ Regular matmul: PASS")
            
            # CUDA benchmark
            cuda_time, cuda_gflops = benchmark(fastpy_module, A, B)
            
            # NumPy benchmark
            numpy_times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = np.matmul(A, B)
                numpy_times.append(time.perf_counter() - start)
            numpy_time = np.mean(numpy_times)
            
            # Handle division by zero for very fast CUDA operations
            if cuda_time < 1e-6:
                speedup = numpy_time / 1e-6
            else:
                speedup = numpy_time / cuda_time
            print(f"  ⚡ Regular CUDA: {cuda_time*1000:.2f}ms, {cuda_gflops:.1f} GFLOPS")
            print(f"  🐍 NumPy: {numpy_time*1000:.2f}ms")
            print(f"  🚀 Speedup: {speedup:.1f}x")
        else:
            print("  ✗ Regular matmul: FAIL")

def main():
    try:
        import fastpy
        print("✓ Successfully imported fastpy extension")

        arr1 = fastpy.fastArray(np.array([[1, 2], [3, 4]]), fastpy.CUDA)
        arr2 = fastpy.fastArray(np.array([[[5, 6], [7, 8]], [[5, 6], [7, 8]]]), fastpy.CUDA)
        arr3 = fastpy.matmul(arr1, arr2)
        print(arr3.to_numpy())

        print("="*50)

        arr4 = fastpy.add(arr1, arr2)
        print(arr4.to_numpy())
        
        # Run tests
        test_standard_matmul(fastpy)
        test_4d_tensors(fastpy)
        
        print("\n✓ All tests completed!")
        
    except ImportError as e:
        print("✗ Failed to import fastpy extension")
        print(e)
        print("Build it first: python build.py")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
