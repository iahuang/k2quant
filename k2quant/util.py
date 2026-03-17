
import numpy as np
import torch
from datasets import load_dataset


def get_calibration_data(
    tokenizer,
    nsamples: int = 256,
    seqlen: int = 4096,
    seed: int = 42,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Load and tokenize calibration data from a HuggingFace text dataset.

    Randomly samples `nsamples` contiguous subsequences of length `seqlen`
    from the concatenated dataset text. Used to compute Hessians and
    calibration activations for quantization.

    Args:
        tokenizer: Any HuggingFace tokenizer (model-agnostic).
        nsamples: Number of calibration sequences to sample.
        seqlen: Token length per sample.
        seed: Random seed for reproducible sample selection.
        dataset_name: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split: Dataset split to use.
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        Token IDs tensor. Shape: (nsamples, seqlen). int64, CPU.
    """
    print(f"  Loading calibration data: {nsamples} samples, seqlen={seqlen}")
    dataset = load_dataset(
        dataset_name, dataset_config, split=split, cache_dir=cache_dir
    )
    all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    print("  Tokenizing...")
    all_tokens = tokenizer(all_text, return_tensors="pt", truncation=False)[
        "input_ids"
    ][0]
    print(f"  Total tokens: {len(all_tokens)}")

    rng = np.random.RandomState(seed)
    max_start = len(all_tokens) - seqlen
    starts = rng.choice(max_start, size=nsamples, replace=False)
    samples = [all_tokens[s : s + seqlen] for s in starts]
    return torch.stack(samples)  # (nsamples, seqlen)


def evaluate_perplexity(
    model,
    tokenizer,
    seqlen: int = 4096,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    device: str = "cuda",
    cache_dir: str | None = None,
    max_chunks: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> float:
    """Evaluate perplexity of a causal language model on a text dataset.

    Splits the test set into non-overlapping chunks of `seqlen` tokens,
    computes cross-entropy loss on each, and returns exp(mean(losses)).

    Args:
        model: Any HuggingFace CausalLM (model-agnostic).
        tokenizer: Corresponding tokenizer.
        seqlen: Evaluation sequence length.
        dataset_name: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split: Dataset split to evaluate on.
        device: Device for evaluation ("cuda", "cpu", etc.).
        cache_dir: Optional HuggingFace cache directory.
        max_chunks: If set, evaluate only this many chunks (for quick tests).
        log_fn: Optional callback for progress logging. Called with strings
            like "Chunk 10/73: PPL = 7.81". Defaults to print().

    Returns:
        Perplexity (float). Lower is better. Computed as
        exp(mean(cross_entropy_per_chunk)).
    """
    if log_fn is None:
        log_fn = print

    dataset = load_dataset(
        dataset_name, dataset_config, split=split, cache_dir=cache_dir
    )
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")["input_ids"][0]
    log_fn(f"  Test tokens: {len(enc)}")

    nlls = []
    n_chunks = len(enc) // seqlen
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    model.eval()

    with torch.no_grad():
        for i in range(n_chunks):
            ids = enc[i * seqlen : (i + 1) * seqlen].unsqueeze(0).to(device)
            loss = model(ids, labels=ids).loss.float().item()
            nlls.append(loss)
            if (i + 1) % 10 == 0 or i == 0:
                log_fn(
                    f"    Chunk {i + 1}/{n_chunks}: PPL = {np.exp(np.mean(nlls)):.2f}"
                )

    return float(np.exp(np.mean(nlls)))
