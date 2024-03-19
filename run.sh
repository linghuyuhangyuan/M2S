CF_RESAMPLE_FLAGS="--t_T 250 --n_sample 1 --t_T_fine 75 --jump_length_coarse 20 --jump_length_fine 10 --jump_n_sample_coarse 10 --jump_n_sample_fine 10 --jump_interval_coarse 15 --jump_interval_fine 10"
BASELINE_RESAMPLE_FLAGS="--jump_length 10 --jump_n_sample 10"
MODEL_FLAGS="--num_heads 1 --attention_resolutions 16 --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--timestep_respacing 270 --ddim_stride 5 --range_t 20 --batch_size 5"
BASIC_CONFIG_FLAGS="--use_inverse_masks False --use_cf True --use_ddim True --special_mask False" 
INPUT_PATH="--base_samples demo/image --mask_path demo/mask/thick"
MODEL_PATH="--model_path_64 models/celebahq64_500000.pt --model_path_256 models/celebahq256_500000.pt"
OUTPUT_PATH="--save_dir results/celebahq/thick"

CUDA_VISIBLE_DEVICES=0 python sample.py $CF_RESAMPLE_FLAGS $BASELINE_RESAMPLE_FLAGS $CLASSIFIER_FLAGS $MODEL_FLAGS $SAMPLE_FLAGS $BASIC_CONFIG_FLAGS $INPUT_PATH $MODEL_PATH $CLASSIFIER_PATH $OUTPUT_PATH