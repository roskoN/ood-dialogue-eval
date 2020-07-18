set -x

python -u ./predict_odin_convai.py --output_dir ./output_dir --model_name_or_path gpt2 --eval_input_file ./data/convai/convai1_data.tsv --init_checkpoint output_dir/GPT2_odin_fix.1e-05.2.1gpu.2020-07-08122108/gpt2_odin_fix_1epoch.pkl
python -u ./predict_odin_convai.py --output_dir ./output_dir --model_name_or_path gpt2 --eval_input_file ./data/convai/convai2_data.tsv --init_checkpoint output_dir/GPT2_odin_fix.1e-05.2.1gpu.2020-07-08122108/gpt2_odin_fix_1epoch.pkl
python -u ./predict_llr_convai.py --output_dir ./output_dir --model_name_or_path gpt2 --eval_input_file ./data/convai/convai1_data.tsv --init_checkpoint_regular output_dir/GPT2_regular.1e-05.16.1gpu.2020-04-17112938/gpt2_regular_1epoch.pkl --init_checkpoint_background output_dir/GPT2_background.1e-05.16.1gpu.2020-04-16123658/GPT2_background_1e.pkl
python -u ./predict_llr_convai.py --output_dir ./output_dir --model_name_or_path gpt2 --eval_input_file ./data/convai/convai2_data.tsv --init_checkpoint_regular output_dir/GPT2_regular.1e-05.16.1gpu.2020-04-17112938/gpt2_regular_1epoch.pkl --init_checkpoint_background output_dir/GPT2_background.1e-05.16.1gpu.2020-04-16123658/GPT2_background_1e.pkl
