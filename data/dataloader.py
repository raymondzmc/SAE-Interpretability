from typing import Any
import math

import einops
import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.config import DataConfig


class StreamingDataLoader(DataLoader):
    """DataLoader that supports __len__ for streaming datasets with known sample count."""
    
    def __init__(self, *args, expected_length: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_length = expected_length
    
    def __len__(self) -> int:
        if self._expected_length is not None:
            return self._expected_length
        else:
            # Fall back to default behavior for non-streaming datasets
            return super().__len__()


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = False,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to
    tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of
    shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if
    parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with
    padding, then remove padding at the end.

    NOTE: Adapted from
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/utils.py#L267
    to handle IterableDataset.

    TODO: Fix typing of tokenizer

    This tokenization is useful for training language models, as it allows us to efficiently train
    on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding).
    Further, for models with absolute positional encodings, this avoids privileging early tokens
    (eg, news articles often begin with CNN, and models may learn to use early positional
    encodings to predict these)

    Args:
        dataset: The dataset to tokenize, assumed to be a HuggingFace text dataset. Can be a regular
            Dataset or an IterableDataset.
        tokenizer: The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        max_length: The length of the context window of the sequence. Defaults to 1024.
        column_name: The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token: Add BOS token at the beginning of each sequence. Defaults to False as this
            is not done during training.

    Returns:
        Dataset or IterableDataset: Returns the tokenized dataset, as a dataset of tensors, with a
        single column called "input_ids".

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it
    just outputs nothing. I'm not super sure why
    """

    # Remove all columns apart from the column_name
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    if tokenizer.pad_token is None:  # pyright: ignore[reportAttributeAccessIssue]
        # We add a padding token, purely to implement the tokenizer. This will be removed before
        # inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # pyright: ignore[reportAttributeAccessIssue]
    # Define the length to chop things up into - leaving space for a bos_token if required
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(
        examples: dict[str, list[str]],
    ) -> dict[
        str,
        NDArray[np.signedinteger[Any]],
    ]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)  # pyright: ignore[reportAttributeAccessIssue]
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses no because HF map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()  # type: ignore
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]  # pyright: ignore[reportAttributeAccessIssue]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)  # pyright: ignore[reportAttributeAccessIssue]
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"input_ids": tokens}

    if isinstance(dataset, IterableDataset):
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name]
        )
    else:
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=[column_name], num_proc=num_proc
        )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def create_dataloaders(
    data_config: DataConfig,
    global_seed: int = 0,
    buffer_size: int = 1000,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and eval DataLoaders with separate splits from simplified config.
    
    Args:
        data_config: The data configuration
        global_seed: Global seed for reproducibility
        buffer_size: Buffer size for streaming datasets
        
    Returns:
        Tuple of (train_loader, eval_loader)
        eval_loader is None if n_eval_samples is None
    """
    # Load and prepare dataset
    dataset = load_dataset(data_config.dataset_name, streaming=data_config.streaming, split=data_config.split)
    seed = data_config.seed if data_config.seed is not None else global_seed
    
    # Shuffle dataset
    if data_config.streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)
    
    # Create train and eval splits
    if data_config.streaming:
        # For streaming datasets, take sequential chunks
        train_dataset = dataset.take(data_config.n_train_samples)
        
        if data_config.n_eval_samples is not None:
            # Skip train samples and take eval samples
            eval_dataset = dataset.skip(data_config.n_train_samples).take(data_config.n_eval_samples)
        else:
            eval_dataset = None
    else:
        # For non-streaming datasets, create proper splits
        total_needed = data_config.n_train_samples + (data_config.n_eval_samples or 0)
        
        # Take only the samples we need to avoid loading the entire dataset
        if hasattr(dataset, '__len__') and len(dataset) > total_needed:
            dataset = dataset.select(range(total_needed))
        
        # Split into train and eval
        train_dataset = dataset.select(range(data_config.n_train_samples))
        
        if data_config.n_eval_samples is not None:
            eval_start = data_config.n_train_samples
            eval_end = eval_start + data_config.n_eval_samples
            eval_dataset = dataset.select(range(eval_start, eval_end))
        else:
            eval_dataset = None
    
    # Process datasets (tokenization if needed)
    if data_config.is_tokenized:
        train_torch_dataset = train_dataset.with_format("torch")
        # Validate tokenization
        sample = next(iter(train_torch_dataset))[data_config.column_name]
        assert isinstance(sample, torch.Tensor) and sample.ndim == 1, "Expected tokenized dataset"
        assert len(sample) == data_config.context_length, f"Expected length {data_config.context_length}, got {len(sample)}"
        
        eval_torch_dataset = eval_dataset.with_format("torch") if eval_dataset is not None else None
    else:
        tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
        train_torch_dataset = tokenize_and_concatenate(
            train_dataset,
            tokenizer,
            max_length=data_config.context_length,
            column_name=data_config.column_name,
            add_bos_token=True,
        )
        
        eval_torch_dataset = None
        if eval_dataset is not None:
            eval_torch_dataset = tokenize_and_concatenate(
                eval_dataset,
                tokenizer,
                max_length=data_config.context_length,
                column_name=data_config.column_name,
                add_bos_token=True,
            )
    
    # Use StreamingDataLoader for streaming datasets, regular DataLoader for others
    if data_config.streaming:
        # Calculate expected number of batches for streaming datasets
        expected_train_batches = math.ceil(data_config.n_train_samples / data_config.train_batch_size)
        train_loader = StreamingDataLoader(
            train_torch_dataset,
            batch_size=data_config.train_batch_size,
            shuffle=False,  # Already shuffled the base dataset
            expected_length=expected_train_batches,
        )
        
        eval_loader = None
        if eval_torch_dataset is not None and data_config.n_eval_samples is not None:
            expected_eval_batches = math.ceil(data_config.n_eval_samples / data_config.eval_batch_size)
            eval_loader = StreamingDataLoader(
                eval_torch_dataset,
                batch_size=data_config.eval_batch_size,
                shuffle=False,
                expected_length=expected_eval_batches,
            )
    else:
        # Use regular DataLoader for non-streaming datasets
        train_loader = DataLoader(
            train_torch_dataset,
            batch_size=data_config.train_batch_size,
            shuffle=False,  # Already shuffled the base dataset
        )

        eval_loader = None
        if eval_torch_dataset is not None:
            eval_loader = DataLoader(
                eval_torch_dataset,
                batch_size=data_config.effective_eval_batch_size,
                shuffle=False,
            )
    
    return train_loader, eval_loader
