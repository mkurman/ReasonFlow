# ReasonFlow 🧠

ReasonFlow is a novel framework designed to implement O(1)-like reasoning capabilities in large language models.  
It uses Multi-Path Generation with Noise to generate and fuse multiple reasoning paths, robustly handling uncertainty.  
This approach improves inference quality, scalability, and generalization for diverse NLP tasks.

![ReasonFlow by mkurman; Grok 3 image](/assets/reasonflow.jpg)

## Multi-Path Generation with Noise

ReasonFlow introduces multiple parallel "thinkers" generating partial outputs or "thoughts" at each step.  

### Core Parameters
• num_of_thinkers: number of parallel reasoning agents  
• num_of_thoughts: tokens each thinker generates before the next iteration  
• topk_thinkers: how many top performers to select from each step  

### Advanced Features
• dynamic_temperature: enables adaptive temperature scheduling based on generation progress  
• diversity_weight: controls the balance between quality and diversity in path selection  
• uncertainty handling: adapts acceptance threshold based on model confidence

### Dynamic Behavior
- Temperature Scheduling: Automatically adjusts sampling temperature based on:
  - Generation progress
  - Model confidence
  - Exploration vs exploitation needs

- Uncertainty Handling: Adapts generation strategy using:
  - Token probability distribution analysis
  - Path diversity measurements
  - Acceptance threshold adjustment

By combining thought paths from the best thinkers and leveraging adaptive mechanisms, ReasonFlow stabilizes overall output. This mechanism allows exploration of multiple reasoning paths and selective fusion of solutions, leading to more coherent and robust results.

## High-Performance Architecture

ReasonFlow is optimized for speed and efficiency with:

- **Efficient Tensor Operations**:
  - Minimal memory transfers using in-place operations
  - Batch processing and operator fusion where possible
  - GPU-resident tensors to avoid CPU-GPU transfers

- **Smart Caching & Prefetching**:
  - Optimized KV-cache handling for iterative generation
  - Specialized fast paths for common generation scenarios
  - LRU caching for repeated module lookups

- **Memory Optimization**:
  - Tensor pre-allocation for predictable memory usage
  - Reduced redundant copies of large embeddings
  - Strategic tensor cleanup based on iteration count

- **Numerical Stability**:
  - Log-domain computations for better precision
  - Weighted token averaging based on thinker quality
  - Improved probability and similarity calculations

The framework's modular design enables efficient scaling to multiple thinkers and long generation sequences while maintaining low latency.

## Installation

1. Clone this repository.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Import and set up the ReasonFlow components:  
   ```python
   from reasonflow import ReasonFlow, ReasonFlowConfig
   
   # Configure with advanced features
   config = ReasonFlowConfig(
       num_of_thinkers=2,
       num_of_thoughts=4,
       topk_thinkers=1,
       dynamic_temperature=True,
       diversity_weight=0.3,
       uncertainty_threshold=0.7
   )
   
   reason_flow = ReasonFlow(config, model, tokenizer)
   ```

2. Generate multiple reasoning paths:
   ```python
   output, best_thinkers_summary = reason_flow.generate_thoughts(
       "Your input here",
       max_new_tokens=128,
       device="cuda"
   )
   ```

### Configuration Options

```python
ReasonFlowConfig(
    num_of_thinkers=2,              # Number of parallel reasoning paths
    num_of_thoughts=4,              # Tokens per generation step
    topk_thinkers=1,                # Number of best paths to select
    acceptance_threshold=0.5,       # Base threshold for path acceptance
    temperatures=[0.7, 1.3],        # Temperature range for sampling
    diversity_weight=0.3,           # Weight for diversity vs quality
    min_temperature=0.5,            # Minimum sampling temperature
    max_temperature=1.5,            # Maximum sampling temperature
    dynamic_temperature=True,       # Enable adaptive temperature
    uncertainty_threshold=0.7,      # Threshold for uncertainty handling
    min_acceptance_threshold=0.3,   # Minimum path acceptance threshold
    max_acceptance_threshold=0.8    # Maximum path acceptance threshold
)
```

### Performance Considerations

For optimal performance:

- **Thinker Count**: For most applications, 2-6 thinkers provide a good balance between quality and speed
- **Batch Size**: Larger thought batches (4-8) can improve throughput on high-end GPUs
- **Device Placement**: Keep tensors on GPU for best performance, avoiding host-device transfers
- **Token vs Hidden States**: Using tokens (`use_tokens=True`) is generally faster than hidden states

Refer to the provided Python files for more detailed implementation insights.
