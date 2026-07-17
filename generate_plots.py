# generate_plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_performance_charts(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # DATA CONFIGURATION (Verified Ryzen CPU Telemetry)
    # ---------------------------------------------------------
    frameworks = ['PyTorch Native (CPU)', 'OpenVINO IR (CPU)']
    latencies = [63.81, 9.80]     # Measured in milliseconds (lower is better)
    throughputs = [15.7, 102.0]   # Measured in Frames Per Second (higher is better)
    
    # Color Palette: Deep Blue/Gray for baseline, Bright Teal for OpenVINO optimized
    colors = ['#4A5568', '#00A3C4']
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # ---------------------------------------------------------
    # CHART 1: INFERENCE LATENCY COMPARISON
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(frameworks, latencies, color=colors, width=0.4, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Inference Latency Drop (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 75)
    
    # Add exact numeric value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 1.5,
                f'{height:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    plt.tight_layout()
    latency_plot_path = os.path.join(output_dir, 'benchmark_latency.png')
    plt.savefig(latency_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Latency comparison plot exported to: {latency_plot_path}")

    # ---------------------------------------------------------
    # CHART 2: THROUGHPUT SPEED COMPARISON
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(frameworks, throughputs, color=colors, width=0.4, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Throughput Speed (FPS)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Frame Throughput Jump (Higher is Better)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 120)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 2.5,
                f'{height:.1f} FPS', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    plt.tight_layout()
    throughput_plot_path = os.path.join(output_dir, 'benchmark_throughput.png')
    plt.savefig(throughput_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Throughput speed comparison plot exported to: {throughput_plot_path}")

if __name__ == "__main__":
    generate_performance_charts()