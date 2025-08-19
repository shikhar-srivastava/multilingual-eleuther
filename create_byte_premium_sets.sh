# Train Split: Take 1 GB * Byte Premium of each monolingual dataset 
# Eval Split: Take last 8,000 lines of each monolingual dataset

python /scratch/ssrivas9/multilingual-eleuther/scripts/create_bp_splits.py \
  --input_root /scratch/ssrivas9/catherinearnett/monolingual_training_data \
  --output_root /scratch/ssrivas9/catherinearnett/monolingual_training_data_bp