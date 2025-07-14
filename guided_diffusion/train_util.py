import copy
import functools
import os
import itertools # Added for chaining iterables
import torch.nn.functional as F # Added for CrossEntropyLoss

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    """
    Manages the training process for a diffusion model, potentially with a caption decoder.

    This loop handles data iteration, forward and backward passes, optimization,
    EMA updates, learning rate annealing, logging, and checkpointing.
    """
    def __init__(
        self,
        *,
        model: th.nn.Module,  # The primary diffusion model (e.g., U-Net)
        classifier: th.nn.Module, # An optional classifier for conditional guidance (expected to be None for this task)
        diffusion, # GaussianDiffusion instance from .gaussian_diffusion
        data: iter, # Data iterator from a DataLoader
        dataloader: th.utils.data.DataLoader, # DataLoader instance itself (for re-initializing data iterator)
        batch_size: int,
        microbatch: int, # Microbatch size for gradient accumulation
        lr: float, # Learning rate
        ema_rate: [float, str], # Exponential Moving Average rate(s) for model parameters
        log_interval: int, # Interval for logging training progress
        save_interval: int, # Interval for saving checkpoints
        resume_checkpoint: str, # Path to a checkpoint to resume training from
        use_fp16: bool = False, # Whether to use mixed-precision training
        fp16_scale_growth: float = 1e-3, # Scale growth factor for FP16 training
        schedule_sampler = None, # Schedule sampler for diffusion timesteps (e.g., UniformSampler, LossAwareSampler)
        weight_decay: float = 0.0, # Weight decay for the optimizer
        lr_anneal_steps: int = 0, # Number of steps for learning rate annealing
        caption_decoder: th.nn.Module = None, # Optional caption decoder model
        caption_loss_lambda: float = 0.1, # Weight for the caption decoder loss
        null_token_idx: int = 0, # Index of the null/padding token for captioning
    ):
        self.model = model
        self.dataloader = dataloader
        self.classifier = classifier # Expected to be None for this specific change detection and captioning task
        self.diffusion = diffusion
        self.caption_decoder = caption_decoder
        self.caption_loss_lambda = caption_loss_lambda
        self.null_token_idx = null_token_idx
        self.data_iter = data # Store the data iterator
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size # If microbatch is not set, use batch_size
        self.lr = lr

        # Parse ema_rate: can be a single float or a comma-separated string of floats
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        optimizer_params = list(self.mp_trainer.master_params)
        if self.caption_decoder is not None:
            optimizer_params.extend(list(self.caption_decoder.parameters()))

        self.opt = AdamW(optimizer_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            ema_master_params_unet = self.mp_trainer.master_params
            if self.caption_decoder is not None:
                ema_master_params_caption_decoder = list(self.caption_decoder.parameters())
                self.ema_params = [
                    copy.deepcopy(list(itertools.chain(ema_master_params_unet, ema_master_params_caption_decoder)))
                    for _ in range(len(self.ema_rate))
                ]
            else:
                self.ema_params = [
                    copy.deepcopy(ema_master_params_unet) for _ in range(len(self.ema_rate))
                ]

        if th.cuda.is_available():
            self.use_ddp = True
            # Initialize DistributedDataParallel (DDP) for the main diffusion model (U-Net)
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()], # Assumes single GPU per process
                output_device=dist_util.dev(), # Place output on the same device
                broadcast_buffers=False, # Don't broadcast buffers (e.g., running means for BatchNorm)
                bucket_cap_mb=128, # Controls the DDP bucketing size for gradient communication
                find_unused_parameters=True, # Useful if parts of the model might not contribute to loss
            )
            # Initialize DDP for the caption decoder model, if provided
            if self.caption_decoder is not None:
                self.ddp_caption_decoder = DDP(
                    self.caption_decoder,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    # find_unused_parameters should be True if some parameters might not receive gradients,
                    # e.g., if the decoder has parts only active during inference or specific conditions.
                    # Set to False if all parameters are guaranteed to be used in loss calculation.
                    find_unused_parameters=False, # Adjust if necessary based on caption_decoder architecture
                )
            else:
                self.ddp_caption_decoder = None # No DDP wrapper if no caption decoder
        else:
            # If CUDA is not available, DDP is not used.
            if dist.get_world_size() > 1:
                # Log a warning if running in a multi-process environment without CUDA.
                logger.warn("Distributed training requires CUDA. Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model # Use the raw model
            self.ddp_caption_decoder = self.caption_decoder # Use the raw caption decoder

    def _load_and_sync_parameters(self):
        """Loads model parameters from a checkpoint and synchronizes them across distributed processes."""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0: # Only rank 0 loads the checkpoint
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                full_state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())

                # Separate state_dict for U-Net (model) and caption_decoder
                # U-Net state_dict: keys not starting with "caption_decoder."
                unet_state_dict = {k: v for k,v in full_state_dict.items() if not k.startswith("caption_decoder.")}
                # strict=False allows loading partial checkpoints or checkpoints with extra keys.
                self.model.load_state_dict(unet_state_dict, strict=False)

                if self.caption_decoder is not None:
                    # Caption decoder state_dict: keys starting with "caption_decoder.", with prefix removed.
                    decoder_state_dict = {
                        k.replace("caption_decoder.", ""): v
                        for k, v in full_state_dict.items() if k.startswith("caption_decoder.")
                    }
                    if decoder_state_dict: # If keys with "caption_decoder." prefix exist
                        self.caption_decoder.load_state_dict(decoder_state_dict, strict=False)
                    else:
                        # Fallback: try loading the full_state_dict directly to caption_decoder.
                        # This might be for older checkpoints where caption_decoder params were not prefixed.
                        # strict=False is important here.
                        logger.log("No 'caption_decoder.' prefixed keys found in checkpoint, attempting to load full state_dict to caption_decoder.")
                        self.caption_decoder.load_state_dict(full_state_dict, strict=False)

        # Synchronize parameters across all processes after loading on rank 0.
        # This ensures all DDP replicas start with the same weights.
        dist_util.sync_params(self.model.parameters())
        if self.caption_decoder is not None:
            dist_util.sync_params(self.caption_decoder.parameters())

    def _load_ema_parameters(self, rate: float):
        """Loads Exponential Moving Average (EMA) parameters from a checkpoint for a given EMA rate."""
        # This list will hold the parameters that EMA tracks.
        # It's a combined list of U-Net's master_params (if using FP16) or regular params,
        # and caption_decoder's parameters.
        ema_params_list = []
        # U-Net parameters (these are master_params if use_fp16 is True)
        ema_params_list.extend(copy.deepcopy(self.mp_trainer.master_params))
        if self.caption_decoder is not None:
            # Caption decoder parameters (assuming they are FP32 and directly tracked by EMA)
            ema_params_list.extend(copy.deepcopy(list(self.caption_decoder.parameters())))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            if dist.get_rank() == 0: # Only rank 0 loads EMA checkpoint
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint} for rate {rate}...")
                ema_state_dict_loaded = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())

                # Apply U-Net EMA state
                # Filter out caption_decoder specific keys
                unet_ema_state_dict = {k: v for k,v in ema_state_dict_loaded.items() if not k.startswith("caption_decoder.")}
                if unet_ema_state_dict:
                    # Convert the U-Net part of the EMA state_dict to the format of master_params
                    unet_master_params_from_ema = self.mp_trainer.state_dict_to_master_params(unet_ema_state_dict)
                    # Copy loaded EMA params to the corresponding part of ema_params_list
                    for i in range(len(self.mp_trainer.master_params)):
                        ema_params_list[i].copy_(unet_master_params_from_ema[i])
                else:
                    logger.log(f"No U-Net EMA parameters found in {ema_checkpoint} for rate {rate}.")


                # Apply Caption Decoder EMA state
                if self.caption_decoder is not None:
                    # Filter for caption_decoder specific keys and remove prefix
                    decoder_ema_state_dict = {
                        k.replace("caption_decoder.", ""): v
                        for k, v in ema_state_dict_loaded.items() if k.startswith("caption_decoder.")
                    }
                    if decoder_ema_state_dict:
                        # The caption_decoder parameters start in ema_params_list after U-Net's parameters.
                        start_idx = len(self.mp_trainer.master_params)
                        # Create a temporary decoder model to load the state_dict into,
                        # then copy its parameters. This handles mapping state_dict keys to parameters correctly.
                        temp_decoder = copy.deepcopy(self.caption_decoder)
                        temp_decoder.load_state_dict(decoder_ema_state_dict)
                        for i, param in enumerate(temp_decoder.parameters()):
                             ema_params_list[start_idx + i].copy_(param.data) # param.data to get the tensor
                    elif not unet_ema_state_dict:
                        # Fallback: If no U-Net EMA keys were found AND no prefixed decoder keys,
                        # try loading the entire ema_state_dict_loaded into the caption_decoder.
                        # This is an edge case, e.g., if an EMA checkpoint *only* contained decoder state
                        # without prefixes (which would be unusual for combined models).
                        logger.log(f"No U-Net specific EMA keys and no 'caption_decoder.' prefixed keys found in {ema_checkpoint}. Attempting to load full EMA state to caption_decoder.")
                        try:
                            temp_decoder = copy.deepcopy(self.caption_decoder)
                            temp_decoder.load_state_dict(ema_state_dict_loaded) # Try loading everything
                            start_idx = len(self.mp_trainer.master_params)
                            for i, param in enumerate(temp_decoder.parameters()):
                                ema_params_list[start_idx + i].copy_(param.data)
                        except RuntimeError as e:
                             logger.warn(f"Could not directly map full EMA state_dict to caption_decoder for rate {rate}. Error: {e}")
                    else:
                        logger.log(f"No caption_decoder EMA parameters found in {ema_checkpoint} for rate {rate} (either prefixed or as fallback).")

        # Synchronize EMA parameters across all DDP processes.
        dist_util.sync_params(ema_params_list)
        return ema_params_list

    def _load_optimizer_state(self):
        """Loads optimizer state from a checkpoint."""
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # Optimizer checkpoint is expected to be in the same directory as the main model checkpoint,
        # with a name like "opt<step>.pt".
        opt_checkpoint_path = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")

        if bf.exists(opt_checkpoint_path):
            if dist.get_rank() == 0: # Only rank 0 loads the optimizer state
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint_path}")
            # Load state dict on the correct device.
            state_dict = dist_util.load_state_dict(opt_checkpoint_path, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
        # Note: Optimizer state is not explicitly synced across processes here.
        # AdamW optimizer state includes running averages (exp_avg, exp_avg_sq) which are computed per-parameter.
        # If training is resumed, these states should ideally be consistent.
        # DDP itself does not synchronize optimizer states. They are managed by each process independently.
        # However, since they are loaded from a checkpoint saved by rank 0, and parameters are synced,
        # subsequent updates should be consistent if the loaded checkpoint was valid.

    def run_loop(self):
        """The main training loop that iterates over the dataset and training steps."""
        # The variable `i` here counts batches processed within the current `run_loop` execution.
        # It's not strictly an epoch counter if `lr_anneal_steps` is the primary termination condition.
        i = 0
        while (
            not self.lr_anneal_steps  # Loop indefinitely if lr_anneal_steps is 0
            or self.step + self.resume_step < self.lr_anneal_steps # Loop until lr_anneal_steps is reached
        ):
            try:
                # Attempt to get the next batch from the data iterator
                full_batch_from_dataloader = next(self.data_iter)
            except StopIteration:
                # If the iterator is exhausted, re-initialize it from the dataloader
                self.data_iter = iter(self.dataloader)
                full_batch_from_dataloader = next(self.data_iter)

            # Execute a single training step with the current batch and empty model_kwargs for diffusion
            # (as classifier-free guidance is typical for this setup, model_kwargs are not used here for diffusion conditionings)
            self.run_step(full_batch_from_dataloader, {})
            
            i += 1 # Increment batch counter for this run_loop session
            if self.step % self.log_interval == 0:
                logger.dumpkvs() # Log key-value pairs
            if self.step % self.save_interval == 0:
                self.save() # Save checkpoint
                # Conditional early exit for testing purposes
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1 # Increment the global training step counter

        # Save the model one last time if it wasn't saved on the final step.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_data: tuple, diffusion_model_kwargs: dict):
        """
        Executes a single training step, including data processing, forward/backward pass, and optimizer step.

        Args:
            batch_data: A tuple containing the batch data from the dataloader.
                        Expected structure: (imgA, imgB, change_mask_gt, _token_all, _token_all_len,
                                           encoded_single_caption, single_caption_length, _name)
            diffusion_model_kwargs: Additional keyword arguments for the diffusion model (typically for conditioning).
                                   Expected to be empty for this unconditional setup.
        """
        # Unpack batch data
        imgA, imgB, change_mask_gt, _token_all, _token_all_len, \
        encoded_single_caption, single_caption_length, _name = batch_data

        # Move image and mask data to the computing device
        imgA_dev = imgA.to(dist_util.dev())
        imgB_dev = imgB.to(dist_util.dev())
        change_mask_gt_dev = change_mask_gt.to(dist_util.dev())

        # Concatenate images (A and B) and the ground truth change mask along the channel dimension.
        # This forms the x_start input for the diffusion model, where the model learns to predict
        # the change_mask_gt given imgA and imgB.
        # Expected shape: [batch_size, num_channels_A + num_channels_B + 1, H, W]
        x_start_for_diffusion_model = th.cat((imgA_dev, imgB_dev, change_mask_gt_dev), dim=1)

        # Move caption data to the computing device
        encoded_captions_target_dev = encoded_single_caption.to(dist_util.dev())
        caption_lengths_target_dev = single_caption_length.to(dist_util.dev())

        # Perform the forward and backward passes
        # Inputs to forward_backward:
        #   - x_start_for_diffusion_model: The combined input (images + GT mask)
        #   - diffusion_model_kwargs: Additional arguments for diffusion (empty here)
        #   - encoded_captions_target_dev: Target encoded captions for the caption decoder loss
        #   - caption_lengths_target_dev: Lengths of the target captions
        _pred_xstart_diffusion = self.forward_backward(
            x_start_for_diffusion_model,
            diffusion_model_kwargs,
            encoded_captions_target_dev,
            caption_lengths_target_dev
        )

        # Perform optimizer step using the mixed-precision trainer
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return _pred_xstart_diffusion

    def forward_backward(self,
                         batch_x_start_diffusion_dev: th.Tensor,
                         diffusion_model_kwargs: dict,
                         batch_encoded_captions_target_dev: th.Tensor,
                         batch_caption_lengths_target_dev: th.Tensor):
        """
        Performs the forward and backward passes for a batch, computing losses and gradients.

        This function handles:
        - Microbatching to manage memory usage.
        - Sampling diffusion timesteps.
        - Computing diffusion model losses using `self.diffusion.training_losses_segmentation`.
        - Computing caption decoder loss if a caption decoder is provided.
        - Summing losses and performing the backward pass using `self.mp_trainer.backward()`.
        - Handling DDP `no_sync` context for gradient accumulation.

        Args:
            batch_x_start_diffusion_dev: The input tensor for the diffusion model (images + GT mask),
                                         already on the correct device.
            diffusion_model_kwargs: Additional keyword arguments for the diffusion model.
            batch_encoded_captions_target_dev: Target encoded captions for the caption decoder,
                                               already on the correct device.
            batch_caption_lengths_target_dev: Lengths of the target captions,
                                             already on the correct device.
        Returns:
            The predicted x_start from the last microbatch of the diffusion model.
        """
        self.mp_trainer.zero_grad() # Zero out gradients before processing the batch
        last_pred_xstart_diffusion_micro = None # To store the prediction from the final microbatch

        # Loop over microbatches
        for i in range(0, batch_x_start_diffusion_dev.shape[0], self.microbatch):
            # Slice the current microbatch from the full batch
            micro_x_start_diffusion = batch_x_start_diffusion_dev[i : i + self.microbatch]
            micro_model_kwargs_diffusion = {
                k: v[i : i + self.microbatch] for k, v in diffusion_model_kwargs.items()
            }
            micro_encoded_captions_target = batch_encoded_captions_target_dev[i : i + self.microbatch]
            micro_caption_lengths_target = batch_caption_lengths_target_dev[i : i + self.microbatch]

            # Determine if this is the last microbatch.
            # This is important for DDP gradient synchronization: gradients are only synced on the last microbatch.
            last_batch_micro = (i + self.microbatch) >= batch_x_start_diffusion_dev.shape[0]

            # Sample diffusion timesteps (t) and corresponding weights for the loss.
            # The schedule_sampler (e.g., UniformSampler, LossAwareSampler) determines how t is chosen.
            t, weights = self.schedule_sampler.sample(micro_x_start_diffusion.shape[0], dist_util.dev())

            # Create a partial function for computing diffusion losses.
            # This pre-fills some arguments to self.diffusion.training_losses_segmentation.
            # self.ddp_model is the DDP-wrapped U-Net.
            # self.classifier is expected to be None.
            compute_losses_diffusion_fn = functools.partial(
                self.diffusion.training_losses_segmentation, # The core loss computation method
                self.ddp_model,  # The DDP-wrapped U-Net model
                self.classifier, # Classifier for guidance (None in this setup)
                micro_x_start_diffusion, # Input to the diffusion model for this microbatch
                t,               # Sampled timesteps
                model_kwargs=micro_model_kwargs_diffusion, # Additional model arguments
            )

            # Compute diffusion losses.
            # If DDP is used and it's not the last microbatch, run under `ddp_model.no_sync()`
            # to accumulate gradients locally without synchronizing them across processes.
            if last_batch_micro or not self.use_ddp:
                terms_diffusion, pred_xstart_diffusion_micro = compute_losses_diffusion_fn()
            else:
                with self.ddp_model.no_sync(): # Accumulate gradients locally
                    terms_diffusion, pred_xstart_diffusion_micro = compute_losses_diffusion_fn()

            # Store the prediction from this microbatch (primarily for returning from the function)
            last_pred_xstart_diffusion_micro = pred_xstart_diffusion_micro

            # Calculate the main diffusion loss component (terms_diffusion["loss"])
            # This usually includes terms like MSE between predicted noise and actual noise.
            diffusion_loss_main_comp = (terms_diffusion["loss"] * weights).mean()

            # Calculate the "cal" loss component (calibration or auxiliary loss for the change map)
            # This is scaled by a factor of 10.0.
            diffusion_cal_loss_comp = terms_diffusion.get("loss_cal", th.tensor(0.0, device=diffusion_loss_main_comp.device))
            diffusion_cal_loss_comp_w = (diffusion_cal_loss_comp * weights).mean() * 10.0 # Weighted and scaled

            # Total diffusion loss is the sum of the main component and the scaled "cal" component.
            diffusion_loss_total_component = diffusion_loss_main_comp + diffusion_cal_loss_comp_w

            # Log all original diffusion loss terms (weighted) for detailed monitoring.
            log_loss_dict(self.diffusion, t, {k_diff: (v_diff * weights) for k_diff, v_diff in terms_diffusion.items()})

            # Initialize caption loss to zero.
            caption_loss = th.tensor(0.0, device=diffusion_loss_total_component.device)

            # Select the caption decoder model to use (DDP-wrapped or raw model).
            caption_decoder_model_to_use = self.ddp_caption_decoder if self.use_ddp else self.caption_decoder

            if caption_decoder_model_to_use is not None:
                # The input to the caption decoder is the predicted x_start (i.e., the refined change map)
                # from the diffusion model for the current microbatch.
                input_to_decoder = pred_xstart_diffusion_micro.detach() # Detach if not backpropping through diffusion model for caption loss

                # Similar to the diffusion model, handle DDP `no_sync` for the caption decoder.
                if self.use_ddp and not last_batch_micro and hasattr(caption_decoder_model_to_use, 'no_sync'):
                    with caption_decoder_model_to_use.no_sync(): # Accumulate gradients locally
                        pred_captions_logits, _, _ = caption_decoder_model_to_use(
                            input_to_decoder,
                            micro_encoded_captions_target,
                            micro_caption_lengths_target.squeeze(-1) # Remove last dim if it's 1
                        )
                else: # Sync gradients or not using DDP
                     pred_captions_logits, _, _ = caption_decoder_model_to_use(
                        input_to_decoder,
                        micro_encoded_captions_target,
                        micro_caption_lengths_target.squeeze(-1) # Remove last dim if it's 1
                    )

                # Prepare logits and targets for cross-entropy loss.
                # Logits: [batch, seq_len, vocab_size] -> [batch*(seq_len-1), vocab_size]
                # Targets: [batch, seq_len] -> [batch*(seq_len-1)]
                # We predict tokens from the second one onwards, so slice off the last logit and first target token.
                active_logits = pred_captions_logits[:, :-1, :].contiguous()
                active_targets = micro_encoded_captions_target[:, 1:].contiguous()

                # Compute cross-entropy loss for caption generation.
                caption_loss = F.cross_entropy(
                    active_logits.view(-1, active_logits.size(-1)), # Flatten logits
                    active_targets.view(-1), # Flatten targets
                    ignore_index=self.null_token_idx # Ignore padding tokens
                )
                logger.logkv_mean("caption_loss", caption_loss.item()) # Log caption loss

            # Combine diffusion loss and caption loss (weighted by lambda).
            micro_total_loss = diffusion_loss_total_component + self.caption_loss_lambda * caption_loss
            logger.logkv_mean("total_loss", micro_total_loss.item()) # Log total loss for the microbatch

            # Perform backward pass using the mixed-precision trainer.
            # This accumulates gradients (scaled if using FP16).
            self.mp_trainer.backward(micro_total_loss)

        return last_pred_xstart_diffusion_micro # Return prediction from the last microbatch

    def _update_ema(self):
        """Updates the Exponential Moving Average (EMA) parameters for both U-Net and caption decoder."""
        # Get current master parameters from the U-Net (via mp_trainer)
        current_master_params_unet = self.mp_trainer.master_params

        if self.caption_decoder is not None:
            # If a caption decoder exists, get its parameters directly.
            # Assumes caption_decoder parameters are not handled by mp_trainer for FP16,
            # and are directly used for EMA.
            current_master_params_caption_decoder = list(self.caption_decoder.parameters())
            # Combine U-Net and caption decoder parameters into a single list for EMA update.
            full_master_params_list = list(itertools.chain(current_master_params_unet, current_master_params_caption_decoder))
        else:
            # If no caption decoder, EMA is only for U-Net parameters.
            full_master_params_list = current_master_params_unet

        # Iterate through each EMA rate and its corresponding set of EMA parameters.
        for rate, ema_param_set in zip(self.ema_rate, self.ema_params):
            update_ema(ema_param_set, full_master_params_list, rate=rate)

    def _anneal_lr(self):
        """Anneals the learning rate based on the number of training steps."""
        if not self.lr_anneal_steps: # If lr_anneal_steps is 0, no annealing is performed.
            return
        # Calculate the fraction of annealing steps completed.
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        # Linearly anneal the learning rate from its initial value (self.lr) down to 0.
        lr = self.lr * (1 - frac_done)
        # Update the learning rate for all parameter groups in the optimizer.
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """Logs training progress (current step and total samples)."""
        logger.logkv("step", self.step + self.resume_step) # Log current global step
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch) # Log total samples processed

    def save(self):
        """
        Saves training checkpoints, including main model, EMA models, and optimizer state.
        Checkpoints are saved only by the rank 0 process in distributed training.
        """
        def get_state_dict_from_params(params_list: list) -> dict:
            """
            Helper function to create a state_dict from a list of EMA parameters,
            handling both U-Net and caption decoder parts.

            Args:
                params_list: A combined list of EMA parameters, starting with U-Net's,
                             followed by caption decoder's (if applicable).
            Returns:
                A state_dict compatible with loading into the combined model.
            """
            # Determine the number of U-Net master parameters (from mp_trainer).
            unet_master_params_count = len(self.mp_trainer.master_params)
            # Extract U-Net EMA parameters from the combined list.
            unet_params_for_statedict = params_list[:unet_master_params_count]
            # Convert U-Net EMA parameters (potentially FP32 master params) to a state_dict (typically FP32 model params).
            unet_state_dict = self.mp_trainer.master_params_to_state_dict(unet_params_for_statedict)

            final_state_dict = unet_state_dict.copy() # Initialize with U-Net state

            if self.caption_decoder is not None:
                # Extract caption decoder EMA parameters from the combined list.
                caption_decoder_params_for_statedict = params_list[unet_master_params_count:]
                caption_decoder_state_dict = {}
                param_idx = 0
                # Map the flat list of decoder EMA params back to named parameters.
                # This relies on the order of parameters in `caption_decoder_params_for_statedict`
                # matching the order from `self.caption_decoder.named_parameters()`.
                temp_decoder_param_names = [name for name, _ in self.caption_decoder.named_parameters()]

                for name in temp_decoder_param_names:
                    if param_idx < len(caption_decoder_params_for_statedict):
                        caption_decoder_state_dict[name] = caption_decoder_params_for_statedict[param_idx]
                        param_idx +=1
                    else:
                        # This warning indicates a mismatch, possibly if the model structure changed
                        # or EMA parameters were not correctly assembled.
                        logger.warn(f"Mismatch in caption decoder EMA param list length while saving parameter: {name}")
                        break

                # Add "caption_decoder." prefix to the keys for the final combined state_dict.
                prefixed_caption_decoder_state_dict = {f"caption_decoder.{k}": v for k,v in caption_decoder_state_dict.items()}
                final_state_dict.update(prefixed_caption_decoder_state_dict)
            return final_state_dict

        # Save main model checkpoint (U-Net + caption decoder)
        if dist.get_rank() == 0: # Only rank 0 process saves checkpoints
            logger.log(f"saving main model (step {self.step+self.resume_step})...")
            # Get U-Net state_dict from master parameters (handles FP16)
            unet_state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
            final_save_state_dict = unet_state_dict.copy() # Start with U-Net parameters

            if self.caption_decoder is not None:
                # Get caption decoder state_dict and add prefix.
                caption_decoder_state_dict = self.caption_decoder.state_dict()
                prefixed_caption_decoder_state_dict = {f"caption_decoder.{k}": v for k,v in caption_decoder_state_dict.items()}
                final_save_state_dict.update(prefixed_caption_decoder_state_dict)

            # Filename includes current global step.
            filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
            filepath = os.path.join(get_blob_logdir(), filename)
            th.save(final_save_state_dict, filepath)
            logger.log(f"Saved main model to {filepath}")

        # Save EMA model checkpoints
        for rate, ema_param_list in zip(self.ema_rate, self.ema_params):
            if dist.get_rank() == 0: # Only rank 0 process saves checkpoints
                logger.log(f"saving EMA model (rate {rate}, step {self.step+self.resume_step})...")
                # Use helper to get combined state_dict from EMA parameters.
                ema_state_dict_to_save = get_state_dict_from_params(ema_param_list)
                # Filename includes EMA rate and current global step.
                filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                filepath = os.path.join(get_blob_logdir(), filename)
                th.save(ema_state_dict_to_save, filepath)
                logger.log(f"Saved EMA model (rate {rate}) to {filepath}")

        # Save optimizer state
        if dist.get_rank() == 0: # Only rank 0 process saves checkpoints
            logger.log(f"saving optimizer state (step {self.step+self.resume_step})...")
            # Filename includes current global step.
            opt_filename = f"optsavedmodel{(self.step+self.resume_step):06d}.pt"
            opt_filepath = os.path.join(get_blob_logdir(), opt_filename)
            th.save(self.opt.state_dict(), opt_filepath)
            logger.log(f"Saved optimizer state to {opt_filepath}")

        dist.barrier() # Ensure all processes wait until rank 0 has finished saving.

def parse_resume_step_from_filename(filename: str) -> int:
    """Parses the training step number from a checkpoint filename."""
    # Example filenames: "savedmodel000100.pt", "emasavedmodel_0.9999_000200.pt"
    # We are interested in the step number like "000100" or "000200".
    # It's usually after "model" or "_rate_".
    parts = filename.split("model")
    if len(parts) < 2:
        # Attempt to parse EMA model filename format if "model" split fails
        parts = filename.split("_") # e.g. ["emasavedmodel", "0.9999", "000200.pt"]
        if len(parts) > 1 and parts[-1].endswith(".pt"):
            numeric_part = parts[-1].replace(".pt", "")
            try:
                return int(numeric_part)
            except ValueError:
                return 0 # Failed to parse step number
        return 0 # Default if parsing fails

    # For "savedmodel<step>.pt"
    step_part = parts[-1].split(".")[0] # Get "000100" from "000100.pt"
    try:
        return int(step_part)
    except ValueError:
        # Handle cases like EMA filenames where step_part might be "_rate_step"
        sub_parts = step_part.split("_") # e.g. ["", "0.9999", "000200"] for EMA names
        if len(sub_parts) > 0:
            try:
                return int(sub_parts[-1]) # Try the last part
            except ValueError:
                return 0 # Failed to parse step number
        return 0 # Failed to parse step number

def get_blob_logdir() -> str:
    """Returns the logging directory managed by the global logger instance."""
    return logger.get_dir()

def find_resume_checkpoint() -> str:
    """
    Placeholder function to find the main resume checkpoint.
    This might involve scanning a directory for the latest checkpoint.
    TODO: Implement if dynamic checkpoint finding is required.
    Currently, resume_checkpoint is passed explicitly.
    """
    return None # Placeholder implementation

def find_ema_checkpoint(main_checkpoint: str, step: int, rate: float) -> str:
    """
    Placeholder function to find an EMA checkpoint corresponding to a main checkpoint, step, and EMA rate.
    It constructs an expected filename and checks for its existence.
    TODO: Implement more robust searching if needed.
    """
    if main_checkpoint is None:
        return None
    # Expected EMA filename format based on the save() method.
    filename = f"emasavedmodel_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts: th.Tensor, losses: dict):
    """
    Logs various components of the diffusion loss, including per-quartile statistics.

    Args:
        diffusion: The GaussianDiffusion instance, used to get `num_timesteps`.
        ts: A tensor of timesteps for the current batch.
        losses: A dictionary where keys are loss names (e.g., "mse", "vb")
                and values are tensors of loss values for the batch.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item()) # Log the mean of the loss component
        # Log per-quartile statistics for more detailed analysis of loss distribution over timesteps
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            # Determine the quartile based on the timestep
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss) # Log mean loss for this quartile
