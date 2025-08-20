#!/usr/bin/env python3
"""
GPU Temperature Testing Script
Finds idle GPUs and runs them at ~50% utilization with 30GB memory allocation
"""

import subprocess
import re
import time
import torch
import torch.cuda as cuda
import numpy as np
from typing import List, Tuple
import signal
import sys
import threading
from datetime import datetime

# Global flag for graceful shutdown
shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutdown signal received. Cleaning up...")
    shutdown_flag.set()
    sys.exit(0)


def get_gpu_info() -> List[Tuple[int, float, float]]:
    """
    Get GPU utilization and memory usage for all GPUs
    Returns: List of tuples (gpu_id, gpu_util_percent, memory_used_mb)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            gpu_id = int(parts[0])
            gpu_util = float(parts[1]) if parts[1] != '[N/A]' else 0.0
            memory_used = float(parts[2]) if parts[2] != '[N/A]' else 0.0
            gpu_info.append((gpu_id, gpu_util, memory_used))
        
        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"Error parsing GPU info: {e}")
        return []


def find_idle_gpus(threshold_util: float = 1.0, threshold_memory: float = 10.0) -> List[int]:
    """
    Find GPUs with utilization <= threshold_util% and memory < threshold_memory MB
    """
    gpu_info = get_gpu_info()
    idle_gpus = []
    
    print(f"\n{'='*60}")
    print(f"GPU Status at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"{'GPU':<5} {'Utilization':<15} {'Memory (MB)':<15} {'Status':<10}")
    print(f"{'-'*60}")
    
    for gpu_id, util, mem in gpu_info:
        is_idle = util <= threshold_util and mem < threshold_memory
        status = "IDLE" if is_idle else "IN USE"
        print(f"{gpu_id:<5} {util:>10.1f}%     {mem:>10.1f}     {status:<10}")
        if is_idle:
            idle_gpus.append(gpu_id)
    
    return idle_gpus


def allocate_memory_and_compute(gpu_id: int, memory_gb: float = 30.0, target_util: float = 0.5):
    """
    Allocate memory on a specific GPU and run computations to maintain target utilization
    """
    try:
        # Set the GPU device
        cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"\n[GPU {gpu_id}] Initializing...")
        
        # Calculate tensor size for memory allocation
        # Each float32 element = 4 bytes
        bytes_per_gb = 1024 * 1024 * 1024
        elements_needed = int((memory_gb * bytes_per_gb) / 4)
        
        # Allocate memory in chunks to avoid single allocation failures
        chunk_size = int(elements_needed / 10)  # Split into 10 chunks
        tensors = []
        
        print(f"[GPU {gpu_id}] Allocating {memory_gb:.1f} GB of memory...")
        for i in range(10):
            try:
                tensor = torch.zeros(chunk_size, dtype=torch.float32, device=device)
                tensors.append(tensor)
            except RuntimeError as e:
                print(f"[GPU {gpu_id}] Warning: Could only allocate {i*3:.1f} GB")
                break
        
        # Create larger computation tensors for more intensive operations
        # Increased matrix sizes for better GPU utilization
        matrix_size = 8192  # Doubled from 4096
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        c = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        
        # Additional tensors for varied operations
        d = torch.randn(matrix_size // 2, matrix_size, device=device, dtype=torch.float32)
        e = torch.randn(matrix_size, matrix_size // 2, device=device, dtype=torch.float32)
        
        print(f"[GPU {gpu_id}] Starting computation loop (target: {target_util*100:.0f}% utilization)...")
        
        # More aggressive duty cycle for 50% utilization
        # Adjust timing based on actual vs target utilization
        compute_duration = 0.7  # Start with 70% active time
        sleep_duration = 0.3    # 30% sleep time
        
        iteration = 0
        last_util_check = time.time()
        adjustment_factor = 1.0
        
        while not shutdown_flag.is_set():
            iteration += 1
            
            # Active computation phase with multiple operations
            start_time = time.time()
            operations_count = 0
            
            while time.time() - start_time < compute_duration * adjustment_factor:
                # Perform various matrix operations for intensive GPU usage
                # Mix of different operations to prevent optimization
                
                # Heavy matrix multiplication
                result1 = torch.matmul(a, b)
                
                # Additional operations for more consistent load
                result2 = torch.matmul(c, result1)
                
                # Element-wise operations
                a = a * 0.9999 + torch.randn_like(a) * 0.0001
                b = b * 0.9999 + torch.randn_like(b) * 0.0001
                
                # More complex operations
                if operations_count % 3 == 0:
                    # Batch matrix multiplication with smaller matrices
                    result3 = torch.matmul(d, e)
                    # SVD or eigenvalue decomposition (very intensive)
                    if operations_count % 9 == 0:
                        try:
                            U, S, V = torch.svd(result3[:1024, :1024])
                        except:
                            pass  # Skip if SVD fails
                
                # Convolution-like operation
                if operations_count % 5 == 0:
                    kernel = torch.randn(64, 64, device=device)
                    conv_result = torch.nn.functional.conv2d(
                        result1[:64, :64].unsqueeze(0).unsqueeze(0),
                        kernel.unsqueeze(0).unsqueeze(0)
                    )
                
                operations_count += 1
                
                # Periodically touch the allocated memory to keep it active
                if iteration % 50 == 0:
                    for tensor in tensors:
                        tensor += 0.00001
                
                # Check for shutdown during computation
                if shutdown_flag.is_set():
                    break
            
            # Sleep phase to control utilization
            if sleep_duration * adjustment_factor > 0 and not shutdown_flag.is_set():
                time.sleep(sleep_duration * adjustment_factor)
            
            # Dynamic adjustment based on measured utilization
            if time.time() - last_util_check > 10:  # Check every 10 seconds
                current_gpu_info = get_gpu_info()
                for gid, util, mem in current_gpu_info:
                    if gid == gpu_id:
                        current_util = util / 100.0
                        # Adjust duty cycle based on current vs target
                        if current_util < target_util - 0.05:  # Below target
                            adjustment_factor = min(1.5, adjustment_factor * 1.1)
                            if iteration % 5 == 0:
                                print(f"[GPU {gpu_id}] Util: {current_util*100:.1f}% - Increasing load (factor: {adjustment_factor:.2f})")
                        elif current_util > target_util + 0.05:  # Above target
                            adjustment_factor = max(0.5, adjustment_factor * 0.9)
                            if iteration % 5 == 0:
                                print(f"[GPU {gpu_id}] Util: {current_util*100:.1f}% - Decreasing load (factor: {adjustment_factor:.2f})")
                        else:
                            if iteration % 5 == 0:
                                print(f"[GPU {gpu_id}] Util: {current_util*100:.1f}% - On target!")
                        break
                last_util_check = time.time()
            
            # Print status every 20 iterations
            if iteration % 20 == 0:
                print(f"[GPU {gpu_id}] Running... Iteration {iteration}, Adjustment: {adjustment_factor:.2f}")
        
        print(f"[GPU {gpu_id}] Shutting down...")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to coordinate GPU testing"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("GPU TEMPERATURE TESTING SCRIPT")
    print("="*70)
    print("This script will:")
    print("  1. Find all idle GPUs (≤1% util, <10MB memory)")
    print("  2. Allocate 30GB memory on each idle GPU")
    print("  3. Run computations to maintain ~50% GPU utilization")
    print("  4. Continue until interrupted (Ctrl+C)")
    print("="*70)
    
    # Find idle GPUs
    idle_gpus = find_idle_gpus()
    
    if not idle_gpus:
        print("\n⚠️  No idle GPUs found!")
        print("All GPUs are currently in use or not available.")
        return
    
    print(f"\n✓ Found {len(idle_gpus)} idle GPU(s): {idle_gpus}")
    
    # Confirm before starting
    response = input("\nStart stress testing on these GPUs? (y/N): ")
    if response.lower() != 'y':
        print("Aborted by user.")
        return
    
    print("\n" + "="*70)
    print("STARTING GPU STRESS TEST")
    print("Press Ctrl+C to stop")
    print("="*70)
    
    # Create threads for each GPU
    threads = []
    for gpu_id in idle_gpus:
        thread = threading.Thread(
            target=allocate_memory_and_compute,
            args=(gpu_id, 30.0, 0.5),  # 30GB memory, 50% utilization
            daemon=True
        )
        thread.start()
        threads.append(thread)
        time.sleep(1)  # Stagger the starts slightly
    
    # Monitor GPUs while running
    try:
        while not shutdown_flag.is_set():
            time.sleep(30)  # Update status every 30 seconds
            if not shutdown_flag.is_set():
                print("\n" + "="*70)
                print(f"STATUS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
                print("="*70)
                current_info = get_gpu_info()
                for gpu_id, util, mem in current_info:
                    if gpu_id in idle_gpus:
                        print(f"GPU {gpu_id}: Utilization={util:>5.1f}%, Memory={mem:>7.1f} MB")
                print("="*70)
    except KeyboardInterrupt:
        pass
    
    print("\nShutting down all GPU threads...")
    shutdown_flag.set()
    
    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=5)
    
    print("✓ All GPU processes stopped.")
    print("Goodbye!")


if __name__ == "__main__":
    # Check if PyTorch CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Please ensure:")
        print("  1. You have an NVIDIA GPU")
        print("  2. NVIDIA drivers are installed")
        print("  3. PyTorch is installed with CUDA support")
        sys.exit(1)
    
    # Run the main program
    main()
