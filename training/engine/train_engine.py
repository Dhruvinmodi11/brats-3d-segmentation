# engine/train_engine.py
#
# Reusable training and validation loops.
# Supports both V1 (single output) and V2 (deep supervision) models.

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.parameters import DEVICE, EPS, LOG_EVERY_BATCHES, ACCUM_STEPS
from core.metrics import compute_combined_segmentation_loss
from core.helpers import (
    SimpleLogger,
    save_checkpoint_atomic,
    _get_rng_state,
    get_gpu_memory_summary_string,
    _format_eta_minutes,
    _epoch_summary_table,
)

DS_WEIGHTS = [1.0, 0.5, 0.25]
BOUNDARY_RAMP_EPOCHS = 20
BOUNDARY_MAX_WEIGHT = 0.5
DSD_ALPHA = 1.0
DSD_TAU = 3.0


def _unpack_seg_output(output, model):
    """Split (logits, ds, dsd?, cls_logits?) from V2 forward; cls_logits last if model.num_cls > 0."""
    num_cls = int(getattr(model, "num_cls", 0) or 0)
    cls_logits = None
    if num_cls > 0 and isinstance(output, tuple) and len(output) >= 2:
        last = output[-1]
        if isinstance(last, torch.Tensor) and last.dim() == 2:
            cls_logits = last
            output = output[:-1]

    if isinstance(output, tuple) and len(output) == 3:
        logits, ds_outputs, dsd_outputs = output
        logits = logits.float()
        ds_outputs = [d.float() for d in ds_outputs]
    elif isinstance(output, tuple) and len(output) == 2:
        logits, ds_outputs = output
        logits = logits.float()
        ds_outputs = [d.float() for d in ds_outputs]
        dsd_outputs = None
    elif isinstance(output, tuple) and len(output) == 1:
        logits = output[0].float()
        ds_outputs = None
        dsd_outputs = None
    else:
        logits = output.float()
        ds_outputs = None
        dsd_outputs = None

    if cls_logits is not None:
        cls_logits = cls_logits.float()
    return logits, ds_outputs, dsd_outputs, cls_logits


def _parse_train_batch(batch_data):
    """Returns images, labels, dist_maps|None, cls_targets|None."""
    if len(batch_data) == 2:
        return batch_data[0], batch_data[1], None, None
    if len(batch_data) == 3:
        a, b, c = batch_data
        # cls: (B,) long; dist_maps: (B, K, D, H, W) float
        if isinstance(c, torch.Tensor) and c.dtype in (torch.int64, torch.long) and c.ndim == 1:
            return a, b, None, c
        return a, b, c, None
    if len(batch_data) == 4:
        return batch_data[0], batch_data[1], batch_data[2], batch_data[3]
    raise ValueError(f"Unexpected batch length {len(batch_data)}")


def _cls_loss_masked(cls_logits, cls_targets, num_cls: int):
    """Cross-entropy ignoring targets < 0."""
    if cls_logits is None or cls_targets is None or num_cls <= 0:
        return torch.tensor(0.0, device=DEVICE)
    valid = cls_targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=cls_logits.device)
    return F.cross_entropy(cls_logits[valid], cls_targets[valid])


def _compute_ds_loss(ds_logits_list, labels, class_weights):
    """Compute deep supervision loss at lower-resolution decoder outputs.

    Each ds output is at a different spatial resolution. We downsample the
    ground truth labels to match, then compute the same combined loss.
    """
    ds_loss = torch.tensor(0.0, device=labels.device)
    for ds_logits, w in zip(ds_logits_list, DS_WEIGHTS):
        target_size = ds_logits.shape[2:]
        if target_size != labels.shape[1:]:
            ds_labels = F.interpolate(
                labels.unsqueeze(1).float(), size=target_size, mode="nearest"
            ).squeeze(1).long()
        else:
            ds_labels = labels
        loss_i, _, _, _ = compute_combined_segmentation_loss(ds_logits, ds_labels, class_weights)
        ds_loss = ds_loss + w * loss_i
    return ds_loss


def _compute_dsd_loss(dsd_outputs):
    """Compute Dual Self-Distillation KL divergence loss.

    dsd_outputs: dict with 'enc_softmax' list and 'dec_softmax' list.
    Encoder: teacher = deepest (smallest), students = shallower (larger).
    Decoder: teacher = shallowest (largest), students = deeper (smaller).
    Teacher is resized to match each student's spatial resolution.
    """
    if dsd_outputs is None:
        return torch.tensor(0.0)

    enc_sm = dsd_outputs.get("enc_softmax", [])
    dec_sm = dsd_outputs.get("dec_softmax", [])
    dsd_loss = torch.tensor(0.0, device=enc_sm[0].device) if enc_sm else torch.tensor(0.0)

    def _kl_at_resolution(teacher, student):
        t = teacher.detach()
        if t.shape[2:] != student.shape[2:]:
            t = F.interpolate(t, size=student.shape[2:], mode="trilinear", align_corners=False)
        t_log = torch.log(t + 1e-8)
        s_log = torch.log(student + 1e-8)
        return (t * (t_log - s_log)).sum(dim=1).mean()

    if len(enc_sm) >= 2:
        teacher_enc = enc_sm[-1]
        for student in enc_sm[:-1]:
            dsd_loss = dsd_loss + DSD_ALPHA * _kl_at_resolution(teacher_enc, student)

    if len(dec_sm) >= 2:
        teacher_dec = dec_sm[0]
        for student in dec_sm[1:]:
            dsd_loss = dsd_loss + DSD_ALPHA * _kl_at_resolution(teacher_dec, student)

    return dsd_loss


def _boundary_weight_for_epoch(epoch):
    """Linear ramp-in from 0 to BOUNDARY_MAX_WEIGHT over BOUNDARY_RAMP_EPOCHS."""
    if epoch >= BOUNDARY_RAMP_EPOCHS:
        return BOUNDARY_MAX_WEIGHT
    return BOUNDARY_MAX_WEIGHT * (epoch / BOUNDARY_RAMP_EPOCHS)


def _compute_dist_maps_gpu(labels):
    """Compute signed distance maps on GPU using CuPy (no CPU/dataloader bottleneck).

    labels: (B, D, H, W) long tensor on CUDA.
    Returns: (B, 3, D, H, W) float32 tensor on same device (3 = foreground classes).
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import distance_transform_edt
    except ImportError as e:
        raise RuntimeError(
            "GPU boundary loss requires CuPy. Install with: pip install cupy-cuda12x"
        ) from e

    # Zero-copy view: torch -> DLPack -> CuPy (same GPU)
    labels_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(labels.contiguous()))
    B, D, H, W = labels_cp.shape
    out = cp.zeros((B, 3, D, H, W), dtype=cp.float32)

    for b in range(B):
        for c in range(3):
            mask = (labels_cp[b] == c + 1)
            if mask.any():
                pos = distance_transform_edt(~mask).astype(cp.float32)
                neg = distance_transform_edt(mask).astype(cp.float32)
                out[b, c] = pos - neg

    # CuPy -> Torch (clone so we own memory after cupy array is freed)
    return torch.from_dlpack(out.toDlpack()).clone()


def run_training_epoch(
    model,
    train_loader,
    optimizer,
    class_weights,
    logger: SimpleLogger,
    global_step: int,
    accum_steps: int = ACCUM_STEPS,
    boundary_weight: float = 0.0,
    use_dsd: bool = False,
    lambda_cls: float = 0.0,
    num_cls: int = 0,
):
    model.train()

    total_loss_sum       = 0.0
    total_dice_proxy_sum = 0.0
    total_ce_sum         = 0.0
    total_bdl_sum        = 0.0
    total_dsd_sum        = 0.0
    total_cls_sum        = 0.0
    total_samples        = 0

    epoch_start       = time.time()
    aborted_early     = False

    progress = tqdm(train_loader, desc="TRAIN", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch_data in enumerate(progress, start=1):
        images, labels, dist_maps, cls_targets = _parse_train_batch(batch_data)
        if dist_maps is not None:
            dist_maps = dist_maps.to(DEVICE, non_blocking=True)
        if cls_targets is not None:
            cls_targets = cls_targets.to(DEVICE, non_blocking=True)

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if boundary_weight > 0 and dist_maps is None:
            dist_maps = _compute_dist_maps_gpu(labels)

        if batch_index == 1 or (batch_index % 200 == 0):
            if not torch.isfinite(images).all():
                logger.log(f"BAD DATA: images contain NaN/Inf at batch {batch_index}")
                aborted_early = True
                break
            label_min = int(labels.min().item())
            label_max = int(labels.max().item())
            if label_min < 0 or label_max > 3:
                logger.log(f"BAD DATA: labels out of range at batch {batch_index}: min={label_min} max={label_max}")
                aborted_early = True
                break

        is_update_step = (batch_index % accum_steps == 0) or (batch_index == len(train_loader))

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)

        logits, ds_outputs, dsd_outputs, cls_logits = _unpack_seg_output(output, model)

        if not torch.isfinite(logits).all():
            logger.log(f"NON-FINITE LOGITS at batch {batch_index} | absmax={float(logits.abs().max().cpu()):.2e}")
            aborted_early = True
            break

        total_loss, dice_loss_value, ce_loss_value, bdl_value = \
            compute_combined_segmentation_loss(
                logits, labels, class_weights,
                boundary_weight=boundary_weight,
                dist_maps=dist_maps,
            )

        if ds_outputs is not None:
            total_loss = total_loss + _compute_ds_loss(ds_outputs, labels, class_weights)

        dsd_loss_value = torch.tensor(0.0, device=logits.device)
        if use_dsd and dsd_outputs is not None:
            dsd_loss_value = _compute_dsd_loss(dsd_outputs)
            total_loss = total_loss + dsd_loss_value

        cls_loss_value = torch.tensor(0.0, device=logits.device)
        if lambda_cls > 0 and num_cls > 0 and cls_logits is not None:
            cls_loss_value = _cls_loss_masked(cls_logits, cls_targets, num_cls)
            total_loss = total_loss + lambda_cls * cls_loss_value

        if not torch.isfinite(total_loss):
            logger.log(
                f"NON-FINITE LOSS at batch {batch_index}: "
                f"loss={float(total_loss.detach().cpu())} "
                f"dice={float(dice_loss_value.detach().cpu())} "
                f"ce={float(ce_loss_value.detach().cpu())} | "
                f"images_absmax={float(images.detach().abs().max().cpu()):.2e} "
                f"logits_absmax={float(logits.detach().abs().max().cpu()):.2e} | "
                f"labels min/max={int(labels.min().item())}/{int(labels.max().item())} | "
                f"{get_gpu_memory_summary_string()}"
            )
            crash_path = logger.log_path.parent / "nan_crash.pt"
            raw_m = getattr(model, "_orig_mod", model)
            save_checkpoint_atomic(crash_path, {
                "batch": batch_index, "global_step": global_step,
                "model": raw_m.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": _get_rng_state(),
            })
            logger.log(f"NaN crash checkpoint saved: {crash_path}")
            aborted_early = True
            break

        (total_loss / accum_steps).backward()
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        batch_size = images.size(0)
        total_loss_sum       += total_loss.item() * batch_size
        total_ce_sum         += ce_loss_value.item() * batch_size
        total_dice_proxy_sum += (1.0 - dice_loss_value.item()) * batch_size
        total_bdl_sum        += bdl_value.item() * batch_size
        total_dsd_sum        += dsd_loss_value.item() * batch_size
        total_cls_sum        += cls_loss_value.item() * batch_size
        total_samples        += batch_size

        elapsed_seconds   = time.time() - epoch_start
        avg_batch_seconds = elapsed_seconds / batch_index
        samples_per_sec   = total_samples / elapsed_seconds if elapsed_seconds > 0 else 0.0
        eta_seconds       = (len(train_loader) - batch_index) * avg_batch_seconds if batch_index >= 20 else 0.0

        if batch_index == 1 or (batch_index % 50 == 0):
            postfix = (
                f"loss={total_loss_sum/total_samples:.4f} "
                f"dice_fg={total_dice_proxy_sum/total_samples:.4f} "
                f"ce={total_ce_sum/total_samples:.4f} "
            )
            if boundary_weight > 0:
                postfix += f"bdl={total_bdl_sum/total_samples:.4f} "
            if use_dsd and total_dsd_sum > 0:
                postfix += f"dsd={total_dsd_sum/total_samples:.4f} "
            if lambda_cls > 0 and num_cls > 0:
                postfix += f"cls={total_cls_sum/total_samples:.4f} "
            postfix += (
                f"{samples_per_sec:.1f}samp/s ETA={_format_eta_minutes(eta_seconds)} "
                f"{get_gpu_memory_summary_string()}"
            )
            progress.set_postfix_str(postfix)

        if batch_index == 1 or (batch_index % LOG_EVERY_BATCHES == 0):
            log_extra = ""
            if boundary_weight > 0:
                log_extra += f" bdl={total_bdl_sum/total_samples:.4f}"
            if use_dsd and total_dsd_sum > 0:
                log_extra += f" dsd={total_dsd_sum/total_samples:.4f}"
            if lambda_cls > 0 and num_cls > 0:
                log_extra += f" cls={total_cls_sum/total_samples:.4f}"
            logger.log(
                f"TRAIN batch {batch_index}/{len(train_loader)} | "
                f"loss={total_loss_sum/total_samples:.4f} dice_fg={total_dice_proxy_sum/total_samples:.4f} "
                f"ce={total_ce_sum/total_samples:.4f}{log_extra} | "
                f"{samples_per_sec:.1f}samp/s ETA={_format_eta_minutes(eta_seconds)} | "
                f"{get_gpu_memory_summary_string()}"
            )

    epoch_seconds = time.time() - epoch_start

    if total_samples == 0:
        return float("nan"), float("nan"), float("nan"), epoch_seconds, global_step, aborted_early

    if aborted_early:
        logger.log(
            f"TRAIN aborted at batch {batch_index}/{len(train_loader)} | "
            f"samples={total_samples} | loss={total_loss_sum/total_samples:.4f}"
        )

    return (
        total_loss_sum / total_samples,
        total_dice_proxy_sum / total_samples,
        total_ce_sum / total_samples,
        epoch_seconds,
        global_step,
        aborted_early,
    )


@torch.no_grad()
def run_validation_epoch(
    model,
    val_loader,
    class_weights,
    logger: SimpleLogger,
    num_cls: int = 0,
):
    model.eval()
    device = next(model.parameters()).device

    total_loss_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_ce_sum   = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_samples  = 0

    cls_correct = 0
    cls_labeled = 0

    inter1 = torch.tensor(0, device=device, dtype=torch.long)
    inter2 = torch.tensor(0, device=device, dtype=torch.long)
    inter3 = torch.tensor(0, device=device, dtype=torch.long)
    pred1  = torch.tensor(0, device=device, dtype=torch.long)
    pred2  = torch.tensor(0, device=device, dtype=torch.long)
    pred3  = torch.tensor(0, device=device, dtype=torch.long)
    true1  = torch.tensor(0, device=device, dtype=torch.long)
    true2  = torch.tensor(0, device=device, dtype=torch.long)
    true3  = torch.tensor(0, device=device, dtype=torch.long)

    epoch_start = time.time()

    progress = tqdm(val_loader, desc="VAL  ", dynamic_ncols=True)

    for batch_index, batch_data in enumerate(progress, start=1):
        images, labels, _, cls_targets = _parse_train_batch(batch_data)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if cls_targets is not None:
            cls_targets = cls_targets.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(images)

        logits, _, _, cls_logits = _unpack_seg_output(output, model)
        logits = logits.float()

        if not torch.isfinite(logits).all():
            logger.log(f"VAL NON-FINITE LOGITS at batch {batch_index}")
            break

        total_loss, _, ce_loss_value, _ = compute_combined_segmentation_loss(logits, labels, class_weights)

        batch_size = images.size(0)
        total_loss_sum += total_loss.detach() * batch_size
        total_ce_sum   += ce_loss_value.detach() * batch_size
        total_samples  += batch_size

        pred_cls = logits.argmax(1)
        p1 = (pred_cls == 1); t1 = (labels == 1)
        p2 = (pred_cls == 2); t2 = (labels == 2)
        p3 = (pred_cls == 3); t3 = (labels == 3)

        inter1 += (p1 & t1).sum(); pred1 += p1.sum(); true1 += t1.sum()
        inter2 += (p2 & t2).sum(); pred2 += p2.sum(); true2 += t2.sum()
        inter3 += (p3 & t3).sum(); pred3 += p3.sum(); true3 += t3.sum()

        if num_cls > 0 and cls_logits is not None and cls_targets is not None:
            valid = cls_targets >= 0
            if valid.any():
                pred_c = cls_logits.argmax(dim=-1)
                cls_correct += (pred_c[valid] == cls_targets[valid]).sum().item()
                cls_labeled += int(valid.sum().item())

        elapsed = time.time() - epoch_start
        avg_batch = elapsed / batch_index
        eta_sec = (len(val_loader) - batch_index) * avg_batch

        if batch_index == 1 or (batch_index % 50 == 0):
            d1 = (2.0 * inter1.float() + EPS) / (pred1.float() + true1.float() + EPS)
            d2 = (2.0 * inter2.float() + EPS) / (pred2.float() + true2.float() + EPS)
            d3 = (2.0 * inter3.float() + EPS) / (pred3.float() + true3.float() + EPS)
            fg = (d1 + d2 + d3) / 3.0
            avg_loss = (total_loss_sum / total_samples).item()
            progress.set_postfix_str(
                f"loss={avg_loss:.4f} mean_fg={fg.item():.4f} "
                f"c1={d1.item():.4f} c2={d2.item():.4f} c3={d3.item():.4f} "
                f"ETA={_format_eta_minutes(eta_sec)} {get_gpu_memory_summary_string()}"
            )

        if batch_index == 1 or (batch_index % LOG_EVERY_BATCHES == 0):
            d1 = (2.0 * inter1.float() + EPS) / (pred1.float() + true1.float() + EPS)
            d2 = (2.0 * inter2.float() + EPS) / (pred2.float() + true2.float() + EPS)
            d3 = (2.0 * inter3.float() + EPS) / (pred3.float() + true3.float() + EPS)
            fg = (d1 + d2 + d3) / 3.0
            avg_loss = (total_loss_sum / total_samples).item()
            logger.log(
                f"VAL batch {batch_index}/{len(val_loader)} | "
                f"loss={avg_loss:.4f} mean_fg={fg.item():.4f} "
                f"c1={d1.item():.4f} c2={d2.item():.4f} c3={d3.item():.4f} | "
                f"ETA={_format_eta_minutes(eta_sec)} | {get_gpu_memory_summary_string()}"
            )

    epoch_seconds = time.time() - epoch_start

    mean_d1 = (2.0 * inter1.float() + EPS) / (pred1.float() + true1.float() + EPS)
    mean_d2 = (2.0 * inter2.float() + EPS) / (pred2.float() + true2.float() + EPS)
    mean_d3 = (2.0 * inter3.float() + EPS) / (pred3.float() + true3.float() + EPS)
    mean_fg = (mean_d1 + mean_d2 + mean_d3) / 3.0

    val_cls_acc = None
    if cls_labeled > 0:
        val_cls_acc = cls_correct / cls_labeled

    return (
        (total_loss_sum / total_samples).item(),
        (total_ce_sum / total_samples).item(),
        mean_d1.item(), mean_d2.item(), mean_d3.item(), mean_fg.item(),
        epoch_seconds,
        val_cls_acc,
    )


def run_training_loop(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    class_weights,
    logger: SimpleLogger,
    ckpt_dir,
    last_ckpt,
    best_ckpt,
    epochs: int,
    start_epoch: int = 1,
    best_mean_fg: float = -1.0,
    global_step: int = 0,
    run_id=None,
    on_epoch_end=None,
    early_stop_patience: int = 10,
    accum_steps: int = ACCUM_STEPS,
    use_boundary_loss: bool = False,
    use_dsd: bool = False,
    lambda_cls: float = 0.0,
    num_cls: int = 0,
    cls_csv_path: str | None = None,
):
    overall_start = time.time()
    label = f" (Run {run_id})" if run_id is not None else ""
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs + 1):
        bdl_w = _boundary_weight_for_epoch(epoch) if use_boundary_loss else 0.0

        header = f"\n{'═' * 60}\n  EPOCH {epoch}/{epochs}{label}"
        if use_boundary_loss:
            header += f"  bdl_w={bdl_w:.3f}"
        header += f"\n{'═' * 60}"
        print(header, flush=True)
        logger._fp.write(header + "\n")

        train_loss, train_dice_fg, _, train_seconds, global_step, aborted = \
            run_training_epoch(
                model=model, train_loader=train_loader,
                optimizer=optimizer, class_weights=class_weights,
                logger=logger, global_step=global_step,
                accum_steps=accum_steps,
                boundary_weight=bdl_w,
                use_dsd=use_dsd,
                lambda_cls=lambda_cls,
                num_cls=num_cls,
            )
        if aborted or not np.isfinite(train_loss):
            logger.log("Train produced non-finite. Stopping.")
            break

        val_loss, _, d1, d2, d3, mean_fg, val_seconds, val_cls_acc = \
            run_validation_epoch(
                model=model, val_loader=val_loader,
                class_weights=class_weights, logger=logger,
                num_cls=num_cls,
            )
        torch.cuda.empty_cache()

        lr_scheduler.step(mean_fg)
        current_lr = optimizer.param_groups[0]["lr"]

        summary = _epoch_summary_table(
            epoch, epochs, train_loss, train_dice_fg,
            val_loss, d1, d2, d3, mean_fg,
            train_seconds / 60, val_seconds / 60, current_lr, best_mean_fg,
        )
        print(summary, flush=True)
        logger._fp.write(summary + "\n")
        logger._fp.flush()

        cls_msg = ""
        if val_cls_acc is not None:
            cls_msg = f" val_cls_acc={val_cls_acc:.4f} (n_labeled_patches)"
        logger.log(
            f"Epoch {epoch} | lr={current_lr:.2e} | "
            f"train loss={train_loss:.4f} dice={train_dice_fg:.4f} ({train_seconds/60:.2f}m) | "
            f"val loss={val_loss:.4f} c1={d1:.4f} c2={d2:.4f} c3={d3:.4f} "
            f"mean_fg={mean_fg:.4f} ({val_seconds/60:.2f}m){cls_msg}"
        )

        raw_model = getattr(model, "_orig_mod", model)
        payload = {
            "epoch": epoch, "global_step": global_step, "best_mean_fg": best_mean_fg,
            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "rng": _get_rng_state(), "lr": current_lr, "val_mean_fg_dice": mean_fg,
            "run_id": run_id,
            "num_cls": int(getattr(raw_model, "num_cls", 0) or 0),
            "model_base": int(getattr(raw_model, "width_base", 0) or 0),
            "lambda_cls": float(lambda_cls),
            "cls_csv_path": cls_csv_path,
        }

        save_checkpoint_atomic(last_ckpt, payload)

        if mean_fg > best_mean_fg:
            best_mean_fg = mean_fg
            payload["best_mean_fg"] = best_mean_fg
            save_checkpoint_atomic(best_ckpt, payload)
            logger.log(f"New best mean_fg={best_mean_fg:.4f} -> saved: {best_ckpt}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.log(f"CUDA peak so far: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")

        if epochs_without_improvement >= early_stop_patience:
            logger.log(
                f"Early stopping: no improvement in mean_fg for {early_stop_patience} epochs. "
                f"Best mean_fg={best_mean_fg:.4f}"
            )
            break

        if on_epoch_end is not None:
            on_epoch_end(epoch)

    return time.time() - overall_start, best_mean_fg
