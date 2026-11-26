import torch
import os
import yaml
import numpy as np
import torch.nn as nn
import argparse
import logging
import sys
from pathlib import Path
import wandb

from torch.utils.tensorboard.writer import SummaryWriter
from cs336_basics.graph import build_shared_pretext_graph, build_shared_context_graph
from cs336_basics.graph_loss import GraphTripletLoss
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module import *
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import *

torch.backends.cudnn.enabled = True
os.environ["ROCM_PATH"] = "/dev/null"

def build_graph_loss(train_data, graph):
    if graph is None or graph.loss_weight == 0:
        return None
    if graph.loss_weight < 0:
        raise ValueError("graph.loss_weight must be nonnegative.")
    builders = {"pretext": build_shared_pretext_graph, "context": build_shared_context_graph}
    if graph.type not in builders:
        raise ValueError("graph.type must be 'pretext' or 'context'.")
    edge_counts = builders[graph.type](
        train_data[:graph.token_limit],
        context_length=graph.context_length,
        max_successors=graph.max_successors,
    )
    return GraphTripletLoss(edge_counts, margin_scale=graph.margin_scale)


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)


def load_config(config_path: str | os.PathLike) -> dict:
    with open(config_path, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    return config


def load_token_prefix(path, limit, context_length):
    """Return the training/evaluation prefix and the total number of tokens on disk."""
    path = Path(path)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"{path}: token limit must be a positive integer or null.")
    if path.suffix == ".bin":
        # The TinyStories preparation script writes one little-endian uint16 per token.
        total, remainder = divmod(path.stat().st_size, 2)
        if remainder:
            raise ValueError(f"{path}: incomplete uint16 token.")
        count = total if limit is None else min(limit, total)
        if count > context_length:
            return np.memmap(path, dtype="<u2", mode="r", shape=(count,)), total
    elif path.suffix == ".npz":
        # Existing experiment configs use compressed NumPy archives.
        with np.load(path) as archive:
            tokens = archive["arr_0"]
        total = len(tokens)
        prefix = tokens[:limit]
        if len(prefix) > context_length:
            return prefix.astype(np.int32, copy=False), total
    else:
        raise ValueError(f"{path}: expected a uint16 .bin file or an .npz archive.")
    raise ValueError(f"{path}: at least {context_length + 1} tokens are needed for a next-token batch.")


def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def set_wandb(config: dict):
    wandb.init(
        project='cs336',
        config=config
    )

def get_tensorboard(save_root) -> SummaryWriter:
    writer = SummaryWriter(os.path.join(save_root, 'tensorboard'))
    return writer

def get_peak_memory(device):
    if device != "cuda":
        return 0

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak_memory


def get_device(config: dict):
    device = "cpu"
    if config.get('device', None) is not None:
        device = config.device
    
    if device == 'cpu':
        return device
    
    if device.startswith('cuda'):
        assert torch.cuda.is_available()
    elif device.startswith('mps'):
        assert hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else:
        raise ValueError('config.device is invalid')

    return device

def get_compile(model: torch.nn.Module, device: str):
    if device.startswith('cuda') or device.startswith('cpu'):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    elif device.startswith('mps'):
        model = torch.compile(model, backend="aot_eager")
    
    return model


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.optimizer.min_lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
        eps=config.optimizer.get('eps', 1e-8))
    return optimizer


def build_tokenizer(config: dict):
    vocab, merges = Tokenizer.load_vocab_merges_from_pkl(
        vocab_filepath = config.tokenizer.vocab_path,
        merges_filepath = config.tokenizer.merges_path
    )
    tokenizer = Tokenizer(vocab=vocab, merges=merges)

    return tokenizer


def build_model(config: dict):
    if config.get('ablation', None) == 'postnorm':
        Module = TransformerLMPostNorm
    elif config.get('ablation', None) == 'norms':
        Module = TransformerLMNoRMS
    elif config.get('ablation', None) == 'nope':
        Module = TransformerLMNoPE
    elif config.get('ablation', None) == 'silu':
        Module = TransformerLMSiLU
    else:
        Module = TransformerLM

    model = Module(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta
    )
    return model


def val_model(model: nn.Module, val_data: np.array, config: dict, device: str):
    model.eval()
    input_batch_data, target_batch_data = get_batch(
        dataset=val_data,
        batch_size=config.trainer.batch_size,
        context_length=config.model.context_length,
        device=device
    )
    with torch.no_grad():
        pred_betch_data = model(input_batch_data)
        val_loss = cross_entropy_loss(inputs=pred_betch_data, targets=target_batch_data)
    model.train()
    return val_loss

def val_model_all(model: nn.Module, val_data: np.array, config: dict, device: str):
    model.eval()
    dataset_len = len(val_data)
    batch_size = config.trainer.batch_size
    context_length = config.model.context_length

    losses = []
    for bs in range((dataset_len - 1) // (batch_size * context_length)):
        start = bs * (batch_size * context_length)
        x = torch.stack([val_data[start + i * context_length : start + (i + 1) * context_length] for i in range(batch_size)], dim=0)
        y = torch.stack([val_data[start + i * context_length + 1 : start + (i + 1) * context_length + 1] for i in range(batch_size)], dim=0)

        x = x.to(device=device).long()
        y = y.to(device=device).long()

        with torch.no_grad():
            pred_y = model(x)
            loss = cross_entropy_loss(inputs=pred_y, targets=y)
            losses.append(loss.cpu().item())

    model.train()
    return sum(losses)/len(losses)

def train(config_path: str, output_dir=None):
    # Log setting.
    exp_name = os.path.basename(config_path).split('.')[0]
    save_root = output_dir or os.path.join('exp_output', exp_name)
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(save_root, 'log.txt')
    logger = get_logger(log_path)
    logger.info(f'The logging will save at {log_path}')

    # Load config file.
    config = Config(load_config(config_path=config_path))
    if config.get('seed') is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    logger.info(f'Loaded configuration from {config_path}')
    # set_wandb(config)
    writer = get_tensorboard(save_root)

    # Get device based on the config.
    device = get_device(config=config)
    logger.info(f'Device: {device}')

    # Build model based on the config.
    model = build_model(config=config)
    model.to(device=device)
    model.train()
    if config.get('compile', True):
        model = get_compile(model, device)

    # Build optimizer based on the config.
    optimizer = build_optimizer(model=model, config=config)

    # Init start iteration and resume if necessary.
    start_iteration = 1
    if config.trainer.get('resume', False) is True:
        logger.info(f"Loading checkpoint from {config.trainer.load_ckpt_path}")
        last_iteration = load_checkpoint(config.trainer.load_ckpt_path, model, optimizer)
        start_iteration = last_iteration + 1
        if start_iteration > config.trainer.max_iteration:
            raise ValueError("Set max_iteration above the checkpoint's completed update count.")
        logger.info(f"Resuming training at iteration {start_iteration}")
    # model = torch.compile(model, backend='aot_eager')

    train_data, train_total = load_token_prefix(
        config.data.train_path, config.data.get("train_token_limit"), config.model.context_length
    )
    valid_data, val_total = load_token_prefix(
        config.data.val_path, config.data.get("val_token_limit"), config.model.context_length
    )
    logger.info(f"Training pool: {len(train_data):,} of {train_total:,} tokens")
    logger.info(f"Validation pool: {len(valid_data):,} of {val_total:,} tokens")
    graph = config.get('graph')
    graph_loss_fn = build_graph_loss(train_data, graph)

    # Start training.
    for it in range(start_iteration, config.trainer.max_iteration + 1):
        # Get batch data.
        input_batch_data, target_batch_data = get_batch(
            dataset=train_data,
            batch_size=config.trainer.batch_size,
            context_length=config.model.context_length,
            device=device
        )

        # Predict data.
        predict_batch_data = model(input_batch_data)

        # Calculate the loss and backward.
        lm_loss = cross_entropy_loss(inputs=predict_batch_data, targets=target_batch_data)
        loss = lm_loss
        if graph_loss_fn is not None:
            graph_loss = graph_loss_fn(
                model.token_embeddings.weight, batch_size=graph.batch_size, num_negatives=graph.num_negatives
            )
            loss = lm_loss + graph.loss_weight * graph_loss
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping.
        l2_norm = gradient_clipping(
            parameters = model.parameters(), 
            max_l2_norm=config.optimizer.max_l2_norm,
        )

        # Update the learning rate.
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=config.optimizer.max_lr,
            min_learning_rate=config.optimizer.min_lr,
            warmup_iters=config.optimizer.warmup_iters,
            cosine_cycle_iters=config.optimizer.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group['lr'] = lr

        # Update the parameters.
        optimizer.step()

        # --------------------------------------------------------------------------------
        # Log the message.
        if it % config.trainer.print_frequency == 0 or it == config.trainer.max_iteration:
            logger.info(f"Iteration: {it}, train loss: {lm_loss.item()}, norm: {l2_norm}")
            # wandb.log({'train_loss': loss.detach().cpu().item()}, step=it)
            writer.add_scalar('train loss', lm_loss.item(), it)
            if graph_loss_fn is not None:
                writer.add_scalar('graph loss', graph_loss.item(), it)
                writer.add_scalar('total loss', loss.item(), it)
            writer.add_scalar('lr', lr, it)
            writer.add_scalar('l2norm', l2_norm, it)

        
        # Evaluation.
        if it % config.trainer.val_frequency == 0 or it == config.trainer.max_iteration:
            val_loss = val_model(model, valid_data, config, device)
            logger.info(f"Iteration: {it}, val loss: {val_loss}")
            # wandb.log({'val_loss': val_loss}, step=it)
            # writer.add_scaler()
            writer.add_scalar('val loss', val_loss, it)


        # Save checkpoint
        if config.trainer.save_frequency and it % config.trainer.save_frequency == 0:
            checkpoint_path = os.path.join(save_root, f"checkpoint_iter_{it}.pt")
            save_checkpoint(model, optimizer, it, checkpoint_path)
            logger.info(f"Saved checkpoint iteration {it} to {checkpoint_path}")
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('cs336: llm-from-zero')
    parser.add_argument('--config', type=str, default='config.yaml', help='the model and trainer config file path')
    parser.add_argument('--output-dir', help='directory for this run’s logs and checkpoints')
    args = parser.parse_args()

    train(args.config, output_dir=args.output_dir)
