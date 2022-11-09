# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python test_intent.py --test_file "${1}" --ckpt_path ./ckpt/intent/best_intent.pt --pred_file "${2}"