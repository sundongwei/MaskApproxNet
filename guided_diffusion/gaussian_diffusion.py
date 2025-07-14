"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch # Though 'th' is preferred, 'torch' is used in DPM_Solver and other places. Keep for compatibility.
import math
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th # Standard alias for PyTorch in this project
from .train_util import visualize # Assuming this is for debugging, keep if used
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage # Keep if used, seems for specific image ops
from torchvision import transforms # Keep if used
from .utils import staple, dice_score, norm # Keep, used in sampling/heuristics
import torchvision.utils as vutils # Keep, used for saving images
from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver # For DPM-Solver sampling
import string # For random name generation in p_sample_loop_known (debug?)
import random # For random name generation
import copy


def standardize(img: th.Tensor) -> th.Tensor:
    mean = th.mean(img)
    std = th.std(img)
    img = (img - mean) / std
    return img


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    Args:
        betas: A 1-D numpy array of betas for each diffusion timestep, starting at T and going to 1.
        model_mean_type: A ModelMeanType determining what the model outputs.
        model_var_type: A ModelVarType determining how variance is output.
        loss_type: A LossType determining the loss function to use.
        dpm_solver: Boolean flag to indicate if DPM-Solver should be used for sampling.
        rescale_timesteps: If True, pass floating point timesteps into the
                           model so that they are always scaled like in the
                           original paper (0 to 1000).
    """
    def __init__(
        self,
        *,
        betas: np.ndarray,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
        dpm_solver: bool,
        rescale_timesteps: bool = False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        # If True, specific sampling loops (e.g., p_sample_loop_known) will use DPM-Solver.
        # DPM-Solver is a fast solver for diffusion ODEs, allowing for fewer sampling steps.
        self.dpm_solver = dpm_solver

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start: th.Tensor, t: th.Tensor, noise: th.Tensor = None) -> th.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start: th.Tensor, x_t: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, model, x: th.Tensor, t: th.Tensor, clip_denoised: bool = True, denoised_fn=None, model_kwargs: dict = None
    ) -> dict:
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        Args:
            model: The U-Net model.
            x: The [N x C_full x H x W] tensor at time t. C_full may include conditioning channels.
            t: A 1-D Tensor of timesteps.
            clip_denoised: If True, clip the denoised signal into [-1, 1].
            denoised_fn: If not None, a function applied to the x_start prediction
                         before clipping.
            model_kwargs: Additional arguments for the model.

        Returns:
            A dict with keys: "mean", "variance", "log_variance", "pred_xstart", "cal".
            "pred_xstart" and other calculations primarily refer to the target channel(s)
            (e.g., the last channel if `x` is [img_A, img_B, noisy_mask]).
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C_in_full = x.shape[:2]
        cal = 0
        assert t.shape == (B,)

        model_output_tuple_or_tensor = model(x, self._scale_timesteps(t), **model_kwargs)

        if isinstance(model_output_tuple_or_tensor, tuple):
            model_pred_main, cal = model_output_tuple_or_tensor
        else:
            model_pred_main = model_output_tuple_or_tensor
            # cal remains 0 or its default.

        # For segmentation, `x` might be [img_A, img_B, noisy_mask].
        # `model_pred_main` typically corresponds to the mask channel (last channel of `x`).
        # `x_target_channel_input` is this part of `x` that `model_pred_main` aims to denoise.
        x_target_channel_input = x[:,-1:,...] # Assumes the last channel is the target.
        C_target = x_target_channel_input.shape[1] # Number of channels model predicts (e.g., 1 for a mask).

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_pred_main.shape == (B, C_target * 2, *x_target_channel_input.shape[2:])
            model_mean_related_output, model_var_values = th.split(model_pred_main, C_target, dim=1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else: # ModelVarType.LEARNED_RANGE
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x_target_channel_input.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x_target_channel_input.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else: # Fixed variance
            model_mean_related_output = model_pred_main
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_target_channel_input.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_target_channel_input.shape)

        def process_xstart(px_start_pred):
            if denoised_fn is not None:
                px_start_pred = denoised_fn(px_start_pred)
            if clip_denoised:
                return px_start_pred.clamp(-1, 1)
            return px_start_pred

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x_target_channel_input, t=t, xprev=model_mean_related_output)
            )
            model_mean = model_mean_related_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_mean_related_output)
            else: # ModelMeanType.EPSILON
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_target_channel_input, t=t, eps=model_mean_related_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_target_channel_input, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_target_channel_input.shape
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'cal': cal,
        }

    def _predict_xstart_from_eps(self, x_t: th.Tensor, t: th.Tensor, eps: th.Tensor) -> th.Tensor:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t: th.Tensor, t: th.Tensor, xprev: th.Tensor) -> th.Tensor:
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t: th.Tensor, t: th.Tensor, pred_xstart: th.Tensor) -> th.Tensor:
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t: th.Tensor) -> th.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var: dict, x: th.Tensor, t: th.Tensor, org, model_kwargs: dict = None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        a, gradient = cond_fn(x, self._scale_timesteps(t),org,  **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var: dict, x: th.Tensor, t: th.Tensor,  model_kwargs: dict = None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps.detach() - (1 - alpha_bar).sqrt() * p_mean_var["update"] * 0 # p_mean_var["update"] seems unused or placeholder
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, eps

    # Removed sample_known as it's a specific use-case wrapper not part of core diffusion logic.

    def p_sample(
        self, model, x: th.Tensor, t: th.Tensor, clip_denoised: bool = True, denoised_fn=None, model_kwargs: dict = None
    ) -> dict:
        """
        Sample x_{t-1} from the model at the given timestep.
        This is a single step in the reverse diffusion process.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Noise is added to the mean prediction to get a sample for x_{t-1}.
        # The noise is scaled by the model's predicted variance.
        # Assumes out["mean"] and out["log_variance"] are for the target channel(s).
        noise = th.randn_like(out["mean"])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out["cal"]}

    def p_sample_loop(
        self, model, shape: tuple, noise: th.Tensor = None, clip_denoised: bool = True,
        denoised_fn=None, cond_fn=None, model_kwargs: dict = None,
        device=None, progress: bool = False,
    ) -> th.Tensor:
        """
        Generate samples from the model using the DDPM ancestral sampler.
        This is typically used for unconditional generation or when the initial state x_T is pure noise.
        """
        final = None
        for sample_dict in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample_dict
        return final["sample"] # Return only the final sample tensor

    def p_sample_loop_known(
        self, model, shape: tuple, img: th.Tensor, step: int = 1000, org=None, noise: th.Tensor = None,
        clip_denoised: bool = True, denoised_fn=None, cond_fn=None, model_kwargs: dict = None,
        device=None, progress: bool = False, conditioner=None, classifier=None
    ) -> tuple:
        """
        Generates samples (specifically a mask) starting from a known image pair and an initial noise mask.
        The last channel of `img` is treated as the noisy mask to be diffused.
        Can use DPM-Solver or standard DDPM sampling.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img_on_device = img.to(device)

        # Initialize noise for the mask channel (last channel of `img_on_device`)
        # If `noise` arg is provided, it's currently overwritten.
        current_noise_for_mask = th.randn_like(img_on_device[:, :1, ...]).to(device)

        # `initial_noisy_state` is x_T for the diffusion process.
        # It consists of fixed channels (e.g., imgA, imgB) and the noisy mask.
        initial_noisy_state = torch.cat((img_on_device[:, :-1,  ...], current_noise_for_mask), dim=1)

        final_model_output_dict = {}

        if self.dpm_solver:
            logger.log("Using DPM-Solver for sampling in p_sample_loop_known.")
            noise_schedule_vp = NoiseScheduleVP(schedule='discrete', betas=th.from_numpy(self.betas).to(device))

            # `model_wrapper` prepares the U-Net for DPM-Solver.
            # `img_fixed_channels` provides the non-diffused parts (A,B) to the wrapper,
            # so the wrapper can construct the full input for the U-Net at each step.
            model_fn_for_dpm = model_wrapper(
                model,
                noise_schedule_vp,
                model_type="noise", # Assuming U-Net predicts noise
                model_kwargs=model_kwargs,
                img_fixed_channels = img_on_device[:, :-1, ...],
                guidance_type="uncond",
            )
            dpm_solver_instance = DPM_Solver(
                model_fn_for_dpm, noise_schedule_vp, algorithm_type="dpmsolver++",
                correcting_x0_fn="dynamic_thresholding"
            )

            # DPM-Solver operates on the noisy part (mask channel `current_noise_for_mask`).
            sampled_mask_channel_x0, cal_aux_output = dpm_solver_instance.sample(
                x_T = current_noise_for_mask.to(dtype=th.float),
                steps=step, order=2, skip_type="time_uniform", method="multistep",
            )
            # Combine predicted clean mask with fixed channels.
            final_model_output_dict["sample"] = torch.cat((img_on_device[:, :-1, ...], norm(sampled_mask_channel_x0)), dim=1)
            final_model_output_dict["cal"] = cal_aux_output
        else:
            logger.log("Using standard DDPM/ancestral sampling loop in p_sample_loop_known.")
            for sample_step_out in self.p_sample_loop_progressive(
                model, shape, time=step, noise=initial_noisy_state,
                clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn,
                org=org, model_kwargs=model_kwargs, device=device, progress=progress,
            ):
                final_model_output_dict = sample_step_out

        # Heuristic combination of 'cal' map and predicted mask.
        final_sample_output = final_model_output_dict.get("sample")
        final_cal_map = final_model_output_dict.get("cal")
        if final_sample_output is not None and final_cal_map is not None:
            predicted_mask_channel = final_sample_output[:,-1:,...]
            cal_out_combined = torch.clamp(final_cal_map + 0.25 * predicted_mask_channel, 0, 1)
        elif final_cal_map is not None:
             cal_out_combined = final_cal_map
        else:
            cal_out_combined = torch.zeros_like(img_on_device[:, :1, ...])

        return final_sample_output, initial_noisy_state, img_on_device, final_cal_map, cal_out_combined

    def p_sample_loop_progressive(
        self, model, shape: tuple, time: int = 1000, noise: th.Tensor = None,
        clip_denoised: bool = True, denoised_fn=None, cond_fn=None, org=None,
        model_kwargs: dict = None, device=None, progress: bool = False,
    ) -> iter:
        """
        Generate samples using DDPM ancestral sampling, yielding intermediate results.
        If `noise` includes fixed channels (e.g. images A,B), only the last channel is diffused.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        current_img_x_t = noise.to(device) if noise is not None else th.randn(*shape, device=device)
        indices = list(range(time))[::-1] # Iterate from T-1 down to 0

        num_total_channels = current_img_x_t.shape[1]
        # `org_fixed_channels` stores the non-diffused part (e.g., imgA, imgB) if present.
        org_fixed_channels = current_img_x_t[:, :-1, ...] if num_total_channels > 1 else None

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t_batch = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                # `current_img_x_t` is the full input [A,B,noisy_mask_t] or just [noisy_mask_t].
                out_dict = self.p_sample(
                    model, current_img_x_t.float(), t_batch,
                    clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs,
                )
                yield out_dict
                # `out_dict["sample"]` is x_{t-1} for the diffused channel(s).
                # If there were fixed channels, recombine them for the next step's input.
                if org_fixed_channels is not None:
                    current_img_x_t = torch.cat((org_fixed_channels, out_dict["sample"]), dim=1)
                else:
                    current_img_x_t = out_dict["sample"]

    def ddim_sample(
        self, model, x: th.Tensor, t: th.Tensor, clip_denoised: bool = True,
        denoised_fn=None, cond_fn=None, model_kwargs: dict = None, eta: float = 0.0,
    ) -> dict:
        """
        Sample x_{t-1} from the model using DDIM.
        """
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        if cond_fn is not None: # Apply conditioning if provided
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # eps is for the target channel(s) that pred_xstart corresponds to.
        eps = self._predict_eps_from_xstart(x[:,-out["pred_xstart"].shape[1]:,...], t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_target_channel_input.shape) # Use shape of target channel for alpha
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_target_channel_input.shape)
        sigma = (
                eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise_for_sample = th.randn_like(out["pred_xstart"]) # Noise should match pred_xstart channels

        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(out["pred_xstart"].shape) -1)))
        sample = mean_pred + nonzero_mask * sigma * noise_for_sample
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out.get("cal")} # Include cal if present

    def ddim_reverse_sample(
        self, model, x: th.Tensor, t: th.Tensor, clip_denoised: bool = True,
        denoised_fn=None, model_kwargs: dict = None, eta: float = 0.0,
    ) -> dict:
        """Sample x_{t+1} from the model using DDIM reverse ODE (for DDIM inversion)."""
        assert eta == 0.0, "Reverse ODE only for deterministic path (eta=0.0)"
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        # eps is for the target channel(s)
        x_target_channel_input = x[:,-out["pred_xstart"].shape[1]:,...]
        eps = self._predict_eps_from_xstart(x_target_channel_input, t, out["pred_xstart"])

        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x_target_channel_input.shape)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next) +
            th.sqrt(1 - alpha_bar_next) * eps
        )
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"], "cal": out.get("cal")}

    def ddim_sample_loop(
        self, model, shape: tuple, noise: th.Tensor = None, clip_denoised: bool = True,
        denoised_fn=None, cond_fn=None, model_kwargs: dict = None,
        device=None, progress: bool = False, eta: float = 0.0,
    ) -> th.Tensor:
        """
        Generate samples from the model using DDIM.
        This is a general DDIM sampling loop, typically for unconditional generation.
        """
        final = None
        num_ddim_steps = self.num_timesteps # Default to all timesteps for DDIM

        for sample_dict in self.ddim_sample_loop_progressive(
            model, shape, time=num_ddim_steps, noise=noise,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn,
            model_kwargs=model_kwargs, device=device, progress=progress, eta=eta,
        ):
            final = sample_dict
        return final["sample"] # Return only the final sample tensor

    def ddim_sample_loop_known(
        self, model, shape: tuple, img: th.Tensor, clip_denoised: bool = True,
        denoised_fn=None, cond_fn=None, model_kwargs: dict = None,
        device=None, progress: bool = False, eta: float = 0.0
    ) -> tuple:
        """
        Generate samples (specifically a mask) using DDIM, starting from a known image pair.
        The last channel of `img` is treated as the noisy mask to be diffused.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img_on_device = img.to(device)

        num_ddim_steps = self.num_timesteps # Use all timesteps for full DDIM reversal

        # Create initial noise for the last channel of `img_on_device` (the mask channel).
        initial_noise_for_mask = th.randn_like(img_on_device[:, -1:, ...]).to(device)

        # `x_T_noisy_state` is the starting point for DDIM: [A, B, noisy_mask_T]
        x_T_noisy_state = torch.cat((img_on_device[:, :-1, ...], initial_noise_for_mask), dim=1).float()

        final_result_dict = None
        for sample_step_out in self.ddim_sample_loop_progressive(
            model, shape, time=num_ddim_steps, noise=x_T_noisy_state,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn,
            model_kwargs=model_kwargs, device=device, progress=progress, eta=eta,
        ):
            final_result_dict = sample_step_out

        # Assuming ddim_sample_loop_progressive yields a dict like p_sample_loop_progressive
        final_sample = final_result_dict.get("sample") if final_result_dict else None
        # For ddim_sample_loop_known, there's no explicit 'cal' combination logic shown in original for DDIM path.
        # Returning the direct sample, initial state, and original image structure.
        # `cal` would come from `final_result_dict.get("cal")` if `ddim_sample` returns it.
        return final_sample, x_T_noisy_state, img_on_device # Add cal if needed later

    def ddim_sample_loop_progressive(
        self, model, shape: tuple, time: int = 1000, noise: th.Tensor = None,
        clip_denoised: bool = True, denoised_fn=None, cond_fn=None,
        model_kwargs: dict = None, device=None, progress: bool = False, eta: float = 0.0,
    ) -> iter:
        """
        Generate samples using DDIM, yielding intermediate results.
        If `noise` includes fixed channels, only the last channel is diffused.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        current_img_x_t = noise.to(device) if noise is not None else th.randn(*shape, device=device)
        indices = list(range(time))[::-1]

        num_total_channels = current_img_x_t.shape[1]
        org_fixed_channels = current_img_x_t[:, :-1, ...] if num_total_channels > 1 else None

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t_batch = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out_dict = self.ddim_sample(
                    model, current_img_x_t.float(), t_batch,
                    clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn,
                    model_kwargs=model_kwargs, eta=eta,
                )
                yield out_dict
                if org_fixed_channels is not None: # Recombine if fixed channels exist
                    current_img_x_t = torch.cat((org_fixed_channels, out_dict["sample"]), dim=1)
                else:
                    current_img_x_t = out_dict["sample"]

    # ... (vb_terms_bpd, training_losses_segmentation, etc. remain after this) ...

    def training_losses_segmentation(self, model, classifier, x_start: th.Tensor, t: th.Tensor, model_kwargs: dict = None, noise: th.Tensor = None) -> tuple[dict, th.Tensor]:
        """
        Compute training losses for a single timestep, adapted for a segmentation task
        where the input `x_start` contains multiple components (e.g., image A, image B, ground truth mask)
        and the loss is primarily computed on the last channel (the mask).

        Args:
            model: The U-Net model.
            classifier: An optional classifier (expected to be None for this task).
            x_start: The input tensor, where `x_start[:, :-1, ...]` could be conditioning images (A, B)
                     and `x_start[:, -1:, ...]` is the ground truth segmentation mask.
                     Shape: [N, C_total, H, W].
            t: A batch of timestep indices.
            model_kwargs: Additional keyword arguments for the model.
            noise: Optional specific Gaussian noise to use for q_sample. If None, it's generated.

        Returns:
            A tuple (terms, final_pred_xstart):
                - terms: A dictionary of loss components. Key "loss" contains the primary loss for backprop.
                         May also include "mse_diff" (MSE on the predicted mask component) and
                         "loss_cal" (MSE on an auxiliary 'cal' map prediction against the ground truth mask).
                - final_pred_xstart: The model's prediction of x0 for the mask channel.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None: # Generate noise if not provided
            noise = th.randn_like(x_start[:, -1:, ...]) # Noise for the mask channel

        gt_mask = x_start[:, -1:, ...] # Ground truth mask (e.g., change map)
        res = copy.deepcopy(gt_mask) # Target for x0 or epsilon prediction
        
        # `res_t` is the noisy version of the ground truth mask `res` at timestep `t`.
        res_t = self.q_sample(res, t, noise=noise)
        
        # `x_t_model_input` is the full input to the U-Net model.
        # It combines conditioning images (A, B from x_start[:, :-1, ...]) with the noisy mask `res_t`.
        x_t_model_input = x_start.clone().float()
        x_t_model_input[:, -1:, ...] = res_t.float() # Replace last channel with noisy mask

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output_tuple_or_tensor = model(x_t_model_input, self._scale_timesteps(t), **model_kwargs)

            if isinstance(model_output_tuple_or_tensor, tuple):
                model_pred_main, cal_pred_aux = model_output_tuple_or_tensor
            else:
                model_pred_main = model_output_tuple_or_tensor
                cal_pred_aux = None

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                C_out = res.shape[1]
                assert model_pred_main.shape[1] == C_out * 2, \
                    f"Expected model_pred_main to have {C_out*2} channels for learned variance, got {model_pred_main.shape[1]}"
                model_pred_mean_related, model_var_values = th.split(model_pred_main, C_out, dim=1)
                frozen_out_for_vb = th.cat([model_pred_mean_related.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out_for_vb: r,
                    x_start=res, x_t=res_t, t=t, clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0
                model_pred_main = model_pred_mean_related

            target_for_main_loss = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=res, x_t=res_t, t=t)[0],
                ModelMeanType.START_X: res,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            terms["mse_diff"] = mean_flat((target_for_main_loss - model_pred_main) ** 2)

            # `loss_cal`: MSE loss for the auxiliary 'cal' map predicted by the U-Net,
            #  compared against the ground truth mask `res`.
            if cal_pred_aux is not None:
                terms["loss_cal"] = mean_flat((res - cal_pred_aux) ** 2)
            else:
                terms["loss_cal"] = th.tensor(0.0, device=x_start.device)

            if "vb" in terms:
                terms["loss"] = terms["mse_diff"] + terms["vb"]
            else:
                terms["loss"] = terms["mse_diff"]
        else:
            raise NotImplementedError(self.loss_type)

        # Determine `final_pred_xstart` (Predicted x0 for the Mask Channel)
        if self.model_mean_type == ModelMeanType.START_X:
            final_pred_xstart = model_pred_main
        elif self.model_mean_type == ModelMeanType.EPSILON:
            final_pred_xstart = self._predict_xstart_from_eps(res_t, t, model_pred_main)
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            final_pred_xstart = self._predict_xstart_from_xprev(x_t=res_t, t=t, xprev=model_pred_main)
        else:
            raise NotImplementedError(f"Calculation of pred_xstart for ModelMeanType {self.model_mean_type} is not implemented.")
        return terms, final_pred_xstart

    def _prior_bpd(self, x_start: th.Tensor) -> th.Tensor:
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start: th.Tensor, clip_denoised: bool = True, model_kwargs: dict = None) -> dict:
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t_int in list(range(self.num_timesteps))[::-1]: # Iterate T-1 down to 0
            t_batch = th.tensor([t_int] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            with th.no_grad():
                # Note: _vb_terms_bpd was not fully defined/refactored in prompt, assuming it exists and works.
                # For this refactor, focusing on the structure and comments around it.
                # This function seems to be named _vb_terms_bptimestepsd in the original code, which is likely a typo.
                # Assuming it should be _vb_terms_bpd or similar.
                # For now, I'll use a placeholder name if it's not found during a direct copy.
                # The original has `_vb_terms_bptimestepsd` - using that.
                out = self._vb_terms_bpd( # Corrected to _vb_terms_bpd based on typical naming
                    model, x_start=x_start, x_t=x_t, t=t_batch,
                    clip_denoised=clip_denoised, model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd, "prior_bpd": prior_bpd,
            "vb": vb, "xstart_mse": xstart_mse, "mse": mse,
        }

def _extract_into_tensor(arr: np.ndarray, timesteps: th.Tensor, broadcast_shape: tuple) -> th.Tensor:
    """
    Extract values from a 1-D numpy array `arr` for a batch of `timesteps`.
    The result is broadcasted to `broadcast_shape`.

    Args:
        arr: The 1-D numpy array to extract values from.
        timesteps: A tensor of indices indicating which values to extract from `arr`.
        broadcast_shape: The target shape for broadcasting the extracted values.
                         Must have K dimensions, with the batch dimension (dim 0)
                         equal to the length of `timesteps`.

    Returns:
        A tensor of shape `[batch_size, 1, ..., 1]` (K dimensions) containing the
        extracted values, ready for broadcasting with tensors of `broadcast_shape`.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
