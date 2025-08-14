import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info


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
    def __init__(self, data, tokenizer, batch_size, max_length, track_bytes=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.track_bytes = track_bytes
        self.byte_tracker = ByteTracker() if track_bytes else None

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

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def get_byte_stats(self) -> dict:
        """Get byte tracking statistics."""
        if self.byte_tracker:
            return self.byte_tracker.get_stats()
        return {"total_bytes": 0, "total_examples": 0, "avg_bytes_per_example": 0}
