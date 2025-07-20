from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DataConfig(BaseModel):
    """Simplified data configuration focusing on key parameters."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    # Dataset configuration
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "gpt2" 
    context_length: int = 256
    
    # Training/evaluation sample counts
    n_train_samples: PositiveInt = Field(..., description="Number of training samples")
    n_eval_samples: PositiveInt | None = Field(None, description="Number of evaluation samples (if None, no evaluation)")
    
    # Data loading parameters
    train_batch_size: PositiveInt = 32
    eval_batch_size: PositiveInt | None = Field(None, description="Eval batch size (if None, uses train_batch_size)")
    streaming: bool = True
    seed: int | None = None
    
    # Advanced options
    is_tokenized: bool = False
    column_name: str = "text"  # Column name in dataset
    split: str = "train"  # Which split to use from dataset
