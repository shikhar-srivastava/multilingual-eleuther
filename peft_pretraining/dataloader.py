import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info

# Import logger for validation and debugging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ByteTracker:
    """
    Tracks UTF-8 byte consumption during dataset processing.
    Thread-safe for use with DataLoader workers.
    """
    def __init__(self):
        self.total_bytes = 0
        self.total_examples = 0
    
    def add_text(self, text: str) -> int:
        """Add text and return its UTF-8 byte count."""
        byte_count = len(text.encode('utf-8'))
        self.total_bytes += byte_count
        self.total_examples += 1
        return byte_count
    
    def get_stats(self) -> dict:
        """Get current tracking statistics."""
        return {
            "total_bytes": self.total_bytes,
            "total_examples": self.total_examples,
            "avg_bytes_per_example": self.total_bytes / max(1, self.total_examples)
        }


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, track_bytes=False, pack_sequences=True, tokenization_batch_size=None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.track_bytes = track_bytes
        self.pack_sequences = pack_sequences
        self.tokenization_batch_size = tokenization_batch_size
        self.byte_tracker = ByteTracker() if track_bytes else None
        
        # For sequence packing
        self.token_buffer = []
        self.attention_buffer = []

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        if self.pack_sequences:
            yield from self._iter_packed(iter_data)
        else:
            yield from self._iter_padded(iter_data)
    
    def _iter_packed(self, iter_data):
        """Iterate with sequence packing for maximum efficiency."""
        batch = []
        batch_bytes = 0
        
        # Reset buffers for this iteration
        self.token_buffer = []
        self.attention_buffer = []
        
        # Collect texts for batch tokenization
        text_batch = []
        
        for example in iter_data:
            text = example["text"].strip()
            if not text:  # Skip empty texts
                continue
                
            if self.track_bytes:
                example_bytes = self.byte_tracker.add_text(text)
                batch_bytes += example_bytes
            
            text_batch.append(text)
            
            # Process batch when we have enough texts (adaptive or user-specified batch size)
            if hasattr(self, 'tokenization_batch_size') and self.tokenization_batch_size:
                batch_threshold = self.tokenization_batch_size
            else:
                batch_threshold = min(32, max(8, 1000 // self.max_length))  # Adaptive: smaller batches for longer sequences
            
            if len(text_batch) >= batch_threshold:
                packed_sequences = self._pack_text_batch(text_batch)
                for seq in packed_sequences:
                    batch.append(seq)
                    if len(batch) == self.batch_size:
                        formatted_batch = self._format_batch(batch)
                        if self.track_bytes:
                            formatted_batch["batch_bytes"] = batch_bytes
                        yield formatted_batch
                        batch = []
                        batch_bytes = 0
                text_batch = []
                
                # Clear token buffer periodically to prevent memory buildup
                if len(self.token_buffer) > self.max_length * 2:
                    logger.debug(f"Clearing oversized token buffer: {len(self.token_buffer)} tokens")
        
        # Process remaining texts
        if text_batch:
            packed_sequences = self._pack_text_batch(text_batch)
            for seq in packed_sequences:
                batch.append(seq)
        
        # Process any remaining buffer content
        if self.token_buffer:
            final_seq = self._create_packed_sequence(
                self.token_buffer[:self.max_length], 
                self.attention_buffer[:self.max_length]
            )
            if final_seq:
                batch.append(final_seq)
        
        # Yield final batch
        if batch:
            formatted_batch = self._format_batch(batch)
            if self.track_bytes:
                formatted_batch["batch_bytes"] = batch_bytes
            yield formatted_batch
    
    def _iter_padded(self, iter_data):
        """Original padded iteration for compatibility."""
        batch = []
        batch_bytes = 0
        
        for example in iter_data:
            # Track UTF-8 bytes before tokenization
            text = example["text"]
            if self.track_bytes:
                example_bytes = self.byte_tracker.add_text(text)
                batch_bytes += example_bytes
            
            tokenized_example = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                formatted_batch = self._format_batch(batch)
                if self.track_bytes:
                    formatted_batch["batch_bytes"] = batch_bytes
                yield formatted_batch
                batch = []
                batch_bytes = 0

        if batch:
            formatted_batch = self._format_batch(batch)
            if self.track_bytes:
                formatted_batch["batch_bytes"] = batch_bytes
            yield formatted_batch

    def _pack_text_batch(self, text_batch):
        """Batch tokenize texts and pack them into sequences.
        Uses overflow windows without overlap and avoids tensorization to handle variable lengths safely.
        """
        if not text_batch:
            return []

        # Batch tokenize for efficiency (2-5x speedup)
        # Use overflow windows with no overlap; do NOT return tensors to avoid shape constraints
        tokenized_batch = self.tokenizer(
            text_batch,
            truncation=True,
            padding=False,
            return_overflowing_tokens=True,
            max_length=self.max_length,
            stride=0,
            return_tensors=None,
        )

        packed_sequences = []

        # Add EOS token between documents for proper separation
        eos_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id

        # tokenized_batch["input_ids"] already contains all windows (including overflow chunks)
        for tokens in tokenized_batch.get("input_ids", []):
            if not tokens:
                continue

            attention = [1] * len(tokens)

            # Add EOS token at the end of each document/window
            if tokens and tokens[-1] != eos_token:
                tokens.append(eos_token)
                attention.append(1)

            # Add to buffer
            self.token_buffer.extend(tokens)
            self.attention_buffer.extend(attention)

            # Create packed sequences when buffer is full
            while len(self.token_buffer) >= self.max_length:
                packed_seq = self._create_packed_sequence(
                    self.token_buffer[:self.max_length],
                    self.attention_buffer[:self.max_length]
                )
                if packed_seq:
                    packed_sequences.append(packed_seq)

                # Keep remainder in buffer
                self.token_buffer = self.token_buffer[self.max_length:]
                self.attention_buffer = self.attention_buffer[self.max_length:]

        return packed_sequences
    
    def _create_packed_sequence(self, token_ids, attention_mask):
        """Create a packed sequence with minimal padding."""
        if not token_ids:
            return None
        
        # Pad to max_length only if needed
        if len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -1
            
            token_ids.extend([pad_token] * pad_length)
            attention_mask.extend([0] * pad_length)  # Padded tokens should be masked
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        }
    
    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        # Validation: Check packing efficiency if enabled
        if self.pack_sequences and len(batch) > 0:
            self._validate_packing_efficiency(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def _validate_packing_efficiency(self, attention_mask):
        """Validate that sequence packing is working effectively."""
        # Calculate padding ratio for this batch
        total_tokens = attention_mask.numel()
        real_tokens = attention_mask.sum().item()
        padding_ratio = 1.0 - (real_tokens / total_tokens)
        
        # Log warning if packing efficiency is poor
        if padding_ratio > 0.3:  # More than 30% padding indicates poor packing
            logger.warning(f"Poor packing efficiency: {padding_ratio:.1%} padding in batch")
        
        # Store efficiency stats for periodic reporting
        if not hasattr(self, '_packing_stats'):
            self._packing_stats = {'total_batches': 0, 'total_padding_ratio': 0.0}
        
        self._packing_stats['total_batches'] += 1
        self._packing_stats['total_padding_ratio'] += padding_ratio
        
        # Report average efficiency every 100 batches
        if self._packing_stats['total_batches'] % 100 == 0:
            avg_padding = self._packing_stats['total_padding_ratio'] / self._packing_stats['total_batches']
            avg_efficiency = 1.0 - avg_padding
            logger.info(f"Packing efficiency over last {self._packing_stats['total_batches']} batches: {avg_efficiency:.1%}")
            # Reset stats
            self._packing_stats = {'total_batches': 0, 'total_padding_ratio': 0.0}
    
    def get_byte_stats(self) -> dict:
        """Get byte tracking statistics."""
        if self.byte_tracker:
            return self.byte_tracker.get_stats()
        return {"total_bytes": 0, "total_examples": 0, "avg_bytes_per_example": 0}


class IntLineIterableDataset(IterableDataset):
    """
    Iterable dataset that reads pre-tokenized files with space-separated integer token IDs (one example per line),
    mirroring the reference approach. Truncates examples to block_size. Shards by rank across distributed nodes.
    
    Args:
        file_path: Path to the pre-tokenized file
        block_size: Maximum sequence length (truncate to this)
        rank: DDP rank for sharding
        world_size: Total number of DDP processes
        vocab_size: If provided, validates that all token IDs are < vocab_size. Lines with
                    out-of-bounds tokens are skipped to prevent CUDA index errors.
    """
    def __init__(self, file_path: str, block_size: int, rank: int = 0, world_size: int = 1, vocab_size: int = None):
        super().__init__()
        self.file_path = file_path
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self._skipped_lines = 0  # Track skipped lines for debugging

    def __iter__(self):
        import itertools
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Shard lines by rank to avoid duplication across DDP ranks
            # Each rank reads lines i where i % world_size == rank
            # Additionally shard across DataLoader workers to prevent duplication when num_workers > 0
            for i, line in enumerate(f):
                if (i % self.world_size) != self.rank:
                    continue
                # Index within this rank's stream
                per_rank_index = i // self.world_size
                if (per_rank_index % num_workers) != worker_id:
                    continue
                s = line.strip()
                if not s:
                    continue
                try:
                    token_ids = [int(x) for x in s.split()]
                except ValueError:
                    # Skip malformed lines
                    continue
                # Validate token IDs are within vocabulary bounds
                if self.vocab_size is not None:
                    max_token = max(token_ids) if token_ids else 0
                    if max_token >= self.vocab_size:
                        self._skipped_lines += 1
                        if self._skipped_lines <= 5:  # Only warn for first few
                            import warnings
                            warnings.warn(
                                f"Skipping line {i+1}: contains token ID {max_token} >= vocab_size {self.vocab_size}. "
                                f"This may indicate a tokenizer/data mismatch."
                            )
                        continue
                if len(token_ids) > self.block_size:
                    token_ids = token_ids[:self.block_size]
                yield {"input_ids": torch.tensor(token_ids, dtype=torch.long)}


def build_intline_collate_fn(pad_token_id: int, max_length: int):
    """
    Returns a collate_fn that pads a batch of variable-length integer sequences to the longest length in the batch
    (bounded by max_length), using pad_token_id (or -1 if pad_token_id is None), and computes attention_mask.
    """
    effective_pad_id = pad_token_id if pad_token_id is not None else -1

    def collate_fn(batch):
        # batch: List[ { "input_ids": Tensor(len_i) } ]
        lengths = [min(len(sample["input_ids"]), max_length) for sample in batch]
        target_len = min(max(lengths), max_length) if lengths else max_length
        padded_inputs = []
        for sample in batch:
            ids = sample["input_ids"][:target_len]
            if ids.size(0) < target_len:
                pad_count = target_len - ids.size(0)
                pad_tensor = torch.full((pad_count,), effective_pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad_tensor], dim=0)
            padded_inputs.append(ids)
        input_ids = torch.stack(padded_inputs, dim=0)
        attention_mask = (input_ids != effective_pad_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate_fn
