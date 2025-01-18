# ReasonFlow 🧠

ReasonFlow is a novel framework designed to implement O(1)-like reasoning capabilities in large language models.  
It uses Multi-Path Generation with Noise to generate and fuse multiple reasoning paths, robustly handling uncertainty.  
This approach improves inference quality, scalability, and generalization for diverse NLP tasks.

## Multi-Path Generation with Noise
ReasonFlow introduces multiple parallel “thinkers” generating partial outputs or “thoughts” at each step.  
• num_of_thinkers: number of parallel reasoning agents.  
• num_of_thoughts: tokens each thinker generates before the next iteration.  
• topk_thinkers: how many top performers to select from each step.  

By combining thought paths from the best thinkers, ReasonFlow stabilizes overall output. This mechanism allows exploration of multiple reasoning paths and selective fusion of solutions, leading to more coherent results.

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
   # ...existing code...
   config = ReasonFlowConfig(num_of_thinkers=2, num_of_thoughts=4, topk_thinkers=1)
   reason_flow = ReasonFlow(config, model, tokenizer)
   ```
2. Generate multiple reasoning paths:
   ```python
   output, best_thinkers_summary = reason_flow.generate_thoughts(
       "Your input here",
       max_new_tokens=128,
       device="cuda"
   )
   # ...existing code...
   ```

Refer to the provided Python files for more details on how ReasonFlow extends GenerationMixin and handles multi-path inference.
