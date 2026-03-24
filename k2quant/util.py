import numpy as np
import torch
from datasets import load_dataset
from typing import Callable


def get_calibration_data(
    tokenizer,
    nsamples: int = 256,
    seqlen: int = 4096,
    seed: int = 42,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
    cache_dir: str | None = None,
) -> torch.Tensor:
    print(f"  Loading calibration data: {nsamples} samples, seqlen={seqlen}")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    print("  Streaming and tokenizing...")
    samples: list[torch.Tensor] = []
    buf_chunks: list[torch.Tensor] = []
    buf_len = 0
    for example in dataset:
        doc = example["text"]
        if not doc.strip():
            continue
        ids = tokenizer(doc, return_tensors="pt", truncation=False)["input_ids"][0]
        buf_chunks.append(ids)
        buf_len += len(ids)

        while buf_len >= seqlen:
            buf = torch.cat(buf_chunks)
            samples.append(buf[:seqlen])
            buf = buf[seqlen:]
            buf_chunks = [buf] if len(buf) > 0 else []
            buf_len = len(buf)
            if len(samples) >= nsamples:
                break

        if len(samples) >= nsamples:
            break

    if len(samples) < nsamples:
        raise ValueError(
            f"Only {len(samples)} sequences could be built, "
            f"but {nsamples} requested. Dataset may be too small."
        )
    print(f"  Collected {len(samples)} sequences")
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
