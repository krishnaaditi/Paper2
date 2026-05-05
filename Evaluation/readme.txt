

*For Amber_Bench Run 
.........................................................................................
.........................................................................................
** For response generation 
python gen_amber_json.py \
  --questions_json data/amber/amber_questions.json \
  --image_root data/coco/val2017 \
  --model_path llava-v1.6-mistral-7b-hf \
  --output_json outputs/amber_llava.json \
  --device cuda:0 \
  --dtype bf16
  
 **For metrics calculation 
 
 python eval_amber_metrics.py ----response {response}
  
  
  
*For MMHal_Bench Run 
.........................................................................................
.........................................................................................
** For response generation 
python gen_mmhal_json.py \
  --questions_jsonl mmhal/mmhal-bench_with_image.jsonl \
  --image_root mmhal/images \
  --model_path \
  --output_jsonl outputs/ \
  --dtype bf16 \
  --max_new_tokens 128

Evaluate on the MMHal Bench
-> python eval_gpt_mmhal.py --response {response} --openai_key {api_key}[cap_file-use response file  generated from inference]

-> python  summarize_gpt_obj_halbench_review.py
 --response {response} --openai_key {api_key}[cap_file-use response file  generated from inference]


*For CHAIR Run 
.........................................................................................
.........................................................................................

** For response generation 
python gen_chair_json.py \
  --image_root data/coco/val2017 \
  --instances_json data/coco/annotations/instances_val2017.json \
  --model_path  \
  --output_json outputs/ \
  --dtype bf16
  
  
   **For metrics calculation 
   
   python eval_chair.py 
  --pred_json outputs ----response {response}
  --instances_json data/coco/annotations/instances_val2017.json
  
  
  
  
  
 ** For causal_hall Run
.........................................................................................
.........................................................................................
** For response generation 

python gen_causal_halbench_json.py \
  --qa_json causal_halbench/qa.json \
  --image_root causal_halbench \
  --model_path  \
  --output_json outputs/ \
  --dtype bf16 \
  --max_new_tokens 16
  
 **For metrics calculation 
  
  python eval_causal_halbench_metrics.py \
  --pred_json outputs/----response {response}
  --qa_json causal_halbench/qa.json
  
  
  
  
  
  
** For MM_spu_Bench Run
.........................................................................................
.........................................................................................
** For response generation 

python gen_mmspubench_json.py \
  --model_path  \
  --output_json outputs/----response {response} \
  --split test \
  --dtype bf16

**For metrics calculation 
  
  python eval_mmspubench_accuracy.py --pred_json outputs//----response {response}
  
  
  
** For POPE Run
.........................................................................................
.........................................................................................
** For response generation 

python gen_pope_json.py \
  --pope_json POPE/output/coco/coco_pope_random.json \
  --image_root data/coco/val2014 \
  --model_path  \
  --output_json outputs/----response {response} \
  --dtype bf16

   
   python gen_pope_json.py \
  --pope_json POPE/output/coco/coco_pope_popular.json \
  --image_root data/coco/val2014 \
  --model_path  \
  --output_json outputs/---response {response} \
  --dtype bf16
  
  
  python gen_pope_json.py \
  --pope_json POPE/output/coco/coco_pope_adversarial.json \
  --image_root data/coco/val2014 \
  --model_path  \
  --output_json outputs/ ---response {response}\
  --dtype bf16
  
    
**For evaluation 

python eval_pope_metrics.py --pred_json outputs/---response {response}

