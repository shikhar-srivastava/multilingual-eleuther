#!/usr/bin/env bash
# Download FineWeb-2 raw text for Amharic, Thai, Urdu, Vietnamese.
# Each block below is independent -- copy and run any single block on its own.
# Amharic and Urdu download everything (small datasets).
# Thai and Vietnamese are capped at 1.02 × BP × eng_ref_bytes to avoid waste.

# ============================================================
# Amharic (fineweb2_amh) — downloads all available (~2.7 GiB)
# ============================================================
source local.env
python scripts/download_fineweb2.py --language amh


# ============================================================
# Thai (fineweb2_tha) — capped download (~88 GiB of ~322 GiB)
# ============================================================
source local.env
ENG_REF="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
python scripts/download_fineweb2.py --language tha \
    --max_bytes $(python3 -c "import math; print(math.floor(1.02 * 2.74 * $ENG_BYTES))")


# ============================================================
# Urdu (fineweb2_urd) — downloads all available (~22.5 GiB)
# ============================================================
source local.env
python scripts/download_fineweb2.py --language urd


# ============================================================
# Vietnamese (fineweb2_vie) — capped download (~43 GiB of ~403 GiB)
# ============================================================
source local.env
ENG_REF="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
ENG_BYTES=$(stat -c%s "$ENG_REF")
python scripts/download_fineweb2.py --language vie \
    --max_bytes $(python3 -c "import math; print(math.floor(1.02 * 1.35 * $ENG_BYTES))")
