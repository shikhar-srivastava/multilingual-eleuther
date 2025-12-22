# Train Split: Take 1 GB * Byte Premium of each monolingual dataset 
# Eval Split: Take last 8,000 lines of each monolingual dataset

python /localdisk/ssrivas9/multilingual-eleuther/scripts/create_bp_splits.py \
  --input_root /localdisk/ssrivas9/catherinearnett/monolingual_training_data \
  --output_root /localdisk/ssrivas9/catherinearnett/monolingual_training_data_bp