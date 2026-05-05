

Download Link for LVLMs
.........................................................

huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf \
  --local-dir /data/Aditi/models/llava-v1.6-mistral-7b-hf

huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /data/Aditi/models/Qwen3-VL-8B-Instruct

huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir /data/Aditi/models/Qwen3.5-9B
  
  
  
  For Training:
..................................................................  
  For VLM model: 
 ..................................................... 
 * only LLM use: train_lora_only.py file 
  *for LLm+projector : train_lora_projector.py file 
  
  
  *For CLIP model:CLIP.py file 
  *For different forge_set ration run: use  train_lora_projector_ratio.py 
  
  
  
 ...................run_scripts...........................
  
  
 For run: use scripts.txt 
 
 

