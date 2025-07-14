"""
Script for generating change segmentation maps and corresponding textual captions
for image pairs using a pre-trained U-Net diffusion model and a caption decoder.

This script performs the following main steps:
1. Loads configuration and sets up distributed processing if applicable.
2. Loads the dataset (WHUCCDataset for change captioning).
3. Loads the pre-trained U-Net model for change map generation via diffusion.
4. Loads the pre-trained caption decoder model.
5. Initializes evaluation metric scorers (BLEU, CIDEr, METEOR, ROUGE).
6. Iterates through the test dataset:
    a. For each image pair, generates a change map using the U-Net model,
       potentially with ensembling.
    b. Generates a descriptive caption for the change using the caption decoder,
       based on the generated change map.
    c. Decodes ground truth captions for evaluation.
    d. Stores generated and ground truth captions.
    e. Saves the generated change map image and caption text.
7. After processing all samples, computes evaluation scores for the generated
   captions against the ground truth captions.
8. Saves all generated captions and evaluation scores to JSON files.

The U-Net model's input channel configuration is critical and must match the
training setup (typically 7 channels: 3 for image A, 3 for image B, and 1 for
an initial noise mask that the diffusion process refines).
"""
import argparse
import os
import json
from collections import OrderedDict
import sys
import random
import time # Keep time import, might be useful for future profiling

import numpy as np
import torch as th
from PIL import Image # Keep PIL for potential future image manipulation, though not directly used now
import torch.distributed as dist
from torchvision.utils import save_image # Explicitly used as vutils.save_image

sys.path.append(".") # Add current directory to sys.path for local module imports

from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.rouge.rouge import Rouge

from guided_diffusion import dist_util, logger
from data.whucc.Whu_CC.whucc import WHUCCDataset # New dataset, old one removed
from module.decoder import DecoderTransformerWithMask

import torchvision.utils as vutils # Used for saving images
from guided_diffusion.utils import staple # staple is used for ensembling
from guided_diffusion.script_util import (
    # NUM_CLASSES, # Might not be used if we are specific to WHUCC, confirmed not used
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Seeding for reproducibility (optional, but good practice)
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed) # Ensure CUDA seeds are also set if using GPU
np.random.seed(seed)
random.seed(seed)

def decode_caption_tokens(tokens: list[int], inv_word_vocab: dict[int, str], word_vocab: dict[str, int]) -> str:
    """
    Decodes a list of token IDs into a human-readable string caption.

    Special tokens like <NULL>, <START>, <END> are removed from the decoded string.

    Args:
        tokens: A list of integer token IDs representing the caption.
        inv_word_vocab: A dictionary mapping token IDs back to words.
        word_vocab: A dictionary mapping words to token IDs (used to identify special tokens).

    Returns:
        A string representing the decoded caption.
    """
    words = [
        inv_word_vocab[token_id] for token_id in tokens
        if token_id not in [word_vocab['<NULL>'], word_vocab['<START>'], word_vocab['<END>']]
    ]
    return " ".join(words)

def main():
    """
    Main function to run the sampling process for change detection and caption generation.
    It loads data, initializes models (U-Net and Caption Decoder), generates change maps
    and captions, and finally evaluates the generated captions against ground truth.
    """
    # --- Setup (Arguments, Distributed, Logger, Output Directory) ---
    args = create_argparser().parse_args()
    dist_util.setup_dist(args) # Setup distributed training environment if applicable
    logger.configure(dir = args.out_dir) # Configure logger to save to output directory

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True) # Create output directory if it doesn't exist
    logger.log(f"Output directory created/specified: {args.out_dir}")

    # --- Data Loading (Dataset, DataLoader, Vocabulary) ---
    logger.log("Creating data loader...")
    dataset = WHUCCDataset( # Use WHUCCDataset for change captioning
        data_folder=args.data_dir,
        list_path=args.list_path_dir,
        split='test', # Inference is typically performed on the test set
        token_folder=args.token_dir, # For loading ground truth captions for evaluation
        vocab_file=args.vocab_name,
        max_length=args.max_caption_length, # Max caption length for dataset processing
        allow_unk=args.allow_unk
    )
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_iter = iter(dataloader) # Create an iterator for the data

    logger.log("Loading vocabulary...")
    vocab_path = os.path.join(args.list_path_dir, args.vocab_name + '.json')
    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f) # Load word -> ID mapping
    vocab_size = len(word_vocab)
    inv_word_vocab = {v: k for k, v in word_vocab.items()} # Create ID -> word mapping
    logger.log(f"Vocabulary loaded with {vocab_size} words.")

    # --- Model Creation and Loading (U-Net & Diffusion, Caption Decoder) ---
    logger.log("Creating U-Net model and diffusion process...")
    unet_model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    # Critical: U-Net `in_channels` (set in create_argparser) must match the pre-trained model's architecture.
    # For this script, it's typically 7 (imgA:3, imgB:3, initial_noise_mask:1).
    # If the U-Net was trained with a different configuration (e.g., 4 channels for difference image + mask),
    # `args.in_channels` and the input preparation (`unet_input_img`) would need to be adjusted.
    # The current script assumes a 7-channel input setup for the U-Net.
    logger.log(f"U-Net model arguments: {unet_model_args}")
    model, diffusion = create_model_and_diffusion(**unet_model_args)

    logger.log(f"Loading U-Net weights from: {args.model_path}")
    unet_state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # Handle potential 'module.' prefix from DistributedDataParallel training
    new_unet_state_dict = OrderedDict()
    for k, v in unet_state_dict.items():
        if k.startswith('module.'): new_unet_state_dict[k[7:]] = v
        else: new_unet_state_dict[k] = v
    if not new_unet_state_dict and unet_state_dict: new_unet_state_dict = unet_state_dict
    model.load_state_dict(new_unet_state_dict)
    model.to(dist_util.dev()) # Move model to the appropriate device (CPU/GPU)
    if args.use_fp16: model.convert_to_fp16() # Convert to FP16 if specified
    model.eval() # Set model to evaluation mode

    logger.log("Creating caption decoder model...")
    caption_decoder = DecoderTransformerWithMask(
        encoder_dim=1, # Input change map from U-Net is single channel
        feature_dim=args.decoder_feature_dim,
        vocab_size=vocab_size,
        max_lengths=args.max_caption_length,
        word_vocab=word_vocab, # Pass the loaded vocabulary
        n_head=args.decoder_n_head,
        n_layers=args.decoder_n_layers,
        dropout=args.decoder_dropout
    )
    if args.caption_decoder_path:
        logger.log(f"Loading caption decoder weights from: {args.caption_decoder_path}")
        cap_dec_state_dict = dist_util.load_state_dict(args.caption_decoder_path, map_location="cpu")
        new_cap_dec_state_dict = OrderedDict()
        for k, v in cap_dec_state_dict.items():
            if k.startswith('module.'): new_cap_dec_state_dict[k[7:]] = v
            else: new_cap_dec_state_dict[k] = v
        if not new_cap_dec_state_dict and cap_dec_state_dict: new_cap_dec_state_dict = cap_dec_state_dict
        caption_decoder.load_state_dict(new_cap_dec_state_dict)
    else:
        logger.log("WARNING: No caption_decoder_path provided. Using initialized caption decoder (random weights).")
    caption_decoder.to(dist_util.dev()) # Move decoder to device
    caption_decoder.eval() # Set decoder to evaluation mode

    # --- Initialize Evaluation Scorers ---
    logger.log("Initializing evaluation scorers...")
    bleu_scorer = Bleu(n=4)
    cider_scorer = Cider()
    meteor_scorer = Meteor() # Ensure METEOR_JAR is set or Java is configured if METEOR encounters issues.
    rouge_scorer = Rouge()

    all_gts = {} # To store ground truth captions for all images
    all_res = {} # To store generated captions for all images

    # --- Main Sampling Loop ---
    logger.log("Starting sampling process...")
    all_generated_captions_details = {} # Stores generated captions with image names
    img_idx = 0 # Counter for processed samples

    for batch_idx, batch_data in enumerate(data_iter):
        # --- Unpack data for the current batch ---
        # WHUCCDataset returns: imgA, imgB, label, token_all, token_all_len, token, token_len, name
        A_img, B_img, _label_placeholder, _token_all, _token_all_len, \
        _encoded_single_caption, _single_caption_length, img_name_tensor = batch_data

        current_img_name_str = img_name_tensor[0] if isinstance(img_name_tensor, (list, tuple)) else img_name_tensor

        # --- Decode ground truth captions for evaluation ---
        gt_captions_for_image = []
        # _token_all shape: [batch_size, num_captions_per_image, max_token_len]
        # _token_all_len shape: [batch_size, num_captions_per_image, 1]
        # Assuming batch_size = 1 for sampling.
        num_gt_captions = _token_all.shape[1]
        for i in range(num_gt_captions):
            gt_len = _token_all_len[0, i, 0].item() # Get length of the i-th GT caption
            if gt_len > 0:
                gt_tokens = _token_all[0, i, :gt_len].tolist() # Get token IDs for the i-th GT caption
                decoded_gt = decode_caption_tokens(gt_tokens, inv_word_vocab, word_vocab)
                if decoded_gt: # Ensure the decoded caption is not an empty string
                    gt_captions_for_image.append(decoded_gt)

        if not gt_captions_for_image:
            logger.log(f"Warning: No ground truth captions found or decoded for image {current_img_name_str}")
            # This image will be skipped by evaluators if it has no GT captions.

        logger.log(f"Processing image: {current_img_name_str} (Batch {batch_idx+1}/{len(dataloader)})")

        # --- Prepare U-Net input: Concatenate image pair and noise for mask ---
        # The U-Net expects a 7-channel input: (imgA (3ch), imgB (3ch), noise_mask (1ch)).
        # The `p_sample_loop_known` function (or DDIM equivalent) internally handles the noise for the mask channel.
        A_img_dev = A_img.to(dist_util.dev())
        B_img_dev = B_img.to(dist_util.dev())
        initial_noise_for_mask = th.randn_like(A_img_dev[:, :1, ...]) # Create a 1-channel noise tensor
        unet_input_img = th.cat((A_img_dev, B_img_dev, initial_noise_for_mask), dim=1) # Shape: [B, 7, H, W]

        # --- Ensemble loop: Generate multiple change maps if num_ensemble > 1 ---
        enslist = [] # List to store generated change maps for ensembling
        logger.log(f"Generating {args.num_ensemble} change map(s) for ensembling...")
        for ens_idx in range(args.num_ensemble):
            model_kwargs = {} # No specific model_kwargs for unconditional generation here
            sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known

            generated_sample, _x_noisy, _org_img_input, _raw_cal, processed_cal_map = sample_fn(
                model,
                (args.batch_size, args.in_channels, args.image_size, args.image_size),
                unet_input_img, # Input to the U-Net for diffusion
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            # `processed_cal_map` is often preferred as the refined change map output.
            enslist.append(processed_cal_map)
            logger.log(f"Generated change map sample {ens_idx+1}/{args.num_ensemble}")

        # --- Ensemble the generated change maps ---
        if args.num_ensemble > 0 and enslist:
            ensembled_change_map = staple(th.stack(enslist, dim=0)).squeeze(0)
            logger.log("Ensembled change maps using STAPLE algorithm.")
        elif enslist: # Should not happen if num_ensemble is at least 1
            ensembled_change_map = enslist[0]
        else:
            logger.log("Warning: No change maps generated for ensembling.")
            ensembled_change_map = None

        if ensembled_change_map is not None:
            # --- Generate caption using the ensembled change map ---
            change_map_for_captioning = ensembled_change_map.to(dist_util.dev())
            logger.log("Generating caption for the ensembled change map...")
            with th.no_grad(): # Disable gradient calculations for inference
                generated_caption_tokens = caption_decoder.sample(
                    encoder_out=change_map_for_captioning, # The change map acts as input to the decoder
                    max_len=args.max_caption_length
                )

            caption_ids_for_first_image = generated_caption_tokens[0] # Assuming batch_size=1
            generated_caption_text = decode_caption_tokens(caption_ids_for_first_image, inv_word_vocab, word_vocab)
            logger.log(f"Image: {current_img_name_str}, Generated Caption: {generated_caption_text}")

            # --- Store generated and ground truth captions for evaluation ---
            all_generated_captions_details[current_img_name_str] = generated_caption_text
            if gt_captions_for_image: # Only store if ground truth is available
                all_gts[current_img_name_str] = gt_captions_for_image
                all_res[current_img_name_str] = [generated_caption_text] # Scorers expect a list of generated captions

            # --- Save artifacts (change map image and caption text) ---
            base_name, _ext = os.path.splitext(current_img_name_str)
            # Use batch_idx for unique filenames if num_samples > dataset size or if names are not unique
            result_filename_png = f"{base_name}_{batch_idx}.png"
            out_path_png = os.path.join(args.out_dir, result_filename_png)
            vutils.save_image(ensembled_change_map, fp=out_path_png, nrow=1, padding=0, normalize=True)
            logger.log(f"Saved change map to {out_path_png}")

            result_filename_txt = f"{base_name}_{batch_idx}_caption.txt"
            out_path_txt = os.path.join(args.out_dir, result_filename_txt)
            with open(out_path_txt, 'w') as f:
                f.write(generated_caption_text)
            logger.log(f"Saved caption to {out_path_txt}")

        img_idx += args.batch_size # Increment total processed sample counter
        if img_idx >= args.num_samples:
            logger.log(f"Reached target number of samples ({args.num_samples}). Stopping.")
            break

    # --- Post-loop: Save all generated captions to a JSON file ---
    logger.log("Sampling complete.")
    all_captions_path = os.path.join(args.out_dir, "_all_generated_captions.json")
    with open(all_captions_path, 'w') as f:
        json.dump(all_generated_captions_details, f, indent=4)
    logger.log(f"Saved all generated captions to {all_captions_path}")

    # --- Evaluation: Compute and log scores (BLEU, CIDEr, METEOR, ROUGE-L) ---
    if not all_gts or not all_res:
        logger.log("No ground truth or generated captions available to evaluate. Skipping evaluation.")
    else:
        logger.log("Computing evaluation scores...")
        eval_img_ids = list(all_gts.keys())

        # Filter gts and res to ensure they only contain common keys and that captions are not empty
        gts_for_bleu = [all_gts[img_id] for img_id in eval_img_ids if img_id in all_res and all_gts[img_id] and all_res[img_id]]
        res_for_bleu = [all_res[img_id] for img_id in eval_img_ids if img_id in all_res and all_gts[img_id] and all_res[img_id]]

        bleu_score_values = None
        if not gts_for_bleu or not res_for_bleu or len(gts_for_bleu) != len(res_for_bleu):
            logger.log("Mismatch or empty lists for Bleu evaluation after filtering. Skipping BLEU.")
        else:
            try:
                bleu_score_val, _ = bleu_scorer.compute_score(gts_for_bleu, res_for_bleu)
                bleu_score_values = bleu_score_val
                logger.log(f"BLEU-1: {bleu_score_values[0]*100:.2f}")
                logger.log(f"BLEU-2: {bleu_score_values[1]*100:.2f}")
                logger.log(f"BLEU-3: {bleu_score_values[2]*100:.2f}")
                logger.log(f"BLEU-4: {bleu_score_values[3]*100:.2f}")
            except Exception as e:
                logger.log(f"Error computing BLEU score: {e}")

        # Prepare data for CIDEr, METEOR, ROUGE (they expect dicts)
        res_for_others = {
            img_id: all_res[img_id]
            for img_id in eval_img_ids
            if img_id in all_res and all_gts[img_id] and all_res[img_id] # Ensure common, non-empty entries
        }
        gts_for_others = {
            img_id: all_gts[img_id]
            for img_id in res_for_others.keys() # Match keys from filtered res_for_others
        }

        cider_score_value, meteor_score_value, rouge_score_value = None, None, None
        if not gts_for_others or not res_for_others:
             logger.log("Mismatch or empty dicts for other evaluations (CIDEr, METEOR, ROUGE). Skipping these.")
        else:
            try:
                cider_score_val, _ = cider_scorer.compute_score(gts_for_others, res_for_others)
                cider_score_value = cider_score_val
                logger.log(f"CIDEr: {cider_score_value*100:.2f}")
            except Exception as e:
                logger.log(f"Error computing CIDEr score: {e}")

            try:
                meteor_score_val, _ = meteor_scorer.compute_score(gts_for_others, res_for_others)
                meteor_score_value = meteor_score_val
                logger.log(f"METEOR: {meteor_score_value*100:.2f}")
            except Exception as e:
                logger.log(f"Error computing METEOR score: {e}. Check METEOR_JAR or Java setup if this persists.")

            try:
                rouge_score_val, _ = rouge_scorer.compute_score(gts_for_others, res_for_others)
                rouge_score_value = rouge_score_val
                logger.log(f"ROUGE-L: {rouge_score_value*100:.2f}")
            except Exception as e:
                logger.log(f"Error computing ROUGE-L score: {e}")

        # --- Save evaluation scores to a JSON file ---
        scores = {
            'BLEU-1': bleu_score_values[0] if bleu_score_values else -1,
            'BLEU-2': bleu_score_values[1] if bleu_score_values else -1,
            'BLEU-3': bleu_score_values[2] if bleu_score_values else -1,
            'BLEU-4': bleu_score_values[3] if bleu_score_values else -1,
            'CIDEr': cider_score_value if cider_score_value is not None else -1,
            'METEOR': meteor_score_value if meteor_score_value is not None else -1,
            'ROUGE-L': rouge_score_value if rouge_score_value is not None else -1,
        }
        scores_path = os.path.join(args.out_dir, "evaluation_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=4)
        logger.log(f"Saved evaluation scores to {scores_path}")

def create_argparser() -> argparse.ArgumentParser:
    """
    Creates and configures the argument parser for the sampling script.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    defaults = dict(
        data_name='whu',  # Dataset identifier (WHUCCDataset is used by default in this script)
        data_dir="./data/whu/Whu_CC/whu_CDC_dataset/images",  # Root image directory for WHUCCDataset
        clip_denoised=True,  # If True, clip the denoised model output x_start to [-1, 1]
        num_samples=10,  # Total number of image pairs to process and generate results for
        batch_size=1,   # Batch size for sampling (must be 1 for current caption generation logic)
        use_ddim=False,  # If True, use DDIM sampler; otherwise, use DDPM sampler
        model_path="",  # Path to the pre-trained U-Net model checkpoint (.pt file)
        caption_decoder_path="",  # Path to the pre-trained caption decoder model checkpoint (.pt file)
        num_ensemble=1,  # Number of samples to average for the change map (ensemble size for STAPLE)
        gpu_dev="0",  # GPU device ID to use (effective if `multi_gpu` is not set or if only one GPU is available)
        out_dir='./results_sampling/',  # Directory to save all outputs (change maps, captions, scores)
        multi_gpu=None,  # For distributed sampling; not commonly used for this script. Set via launch script.

        # WHUCCDataset specific arguments
        list_path_dir='./data/whu/Whu_CC/',  # Directory containing train/val/test list files and vocab.json
        token_dir='./data/whu/Whu_CC/tokens/',  # Directory containing tokenized ground truth caption files
        vocab_name='vocab',  # Base name of the vocabulary file (expected: vocab.json)
        allow_unk=1,  # Whether to allow unknown tokens when loading dataset captions (1=True, 0=False)

        # DecoderTransformerWithMask specific arguments for the caption decoder
        max_caption_length=40,  # Maximum length for generated captions; also used by dataset for padding
        decoder_feature_dim=512,  # Feature dimension for the caption decoder's transformer layers
        decoder_n_head=8,  # Number of attention heads in the caption decoder's transformer
        decoder_n_layers=6,  # Number of layers in the caption decoder's transformer
        decoder_dropout=0.1,  # Dropout rate used in the caption decoder's transformer
    )
    defaults.update(model_and_diffusion_defaults()) # Add default model and diffusion arguments

    # Critical U-Net input channel configuration:
    # This setting must match the architecture of the pre-trained U-Net model specified by `model_path`.
    # The typical setup for this script is 7 channels:
    # - 3 channels for image A (e.g., RGB)
    # - 3 channels for image B (e.g., RGB)
    # - 1 channel for the initial noise mask that the diffusion model refines into a change map.
    # If the U-Net was trained with a different input structure (e.g., 4 channels for a difference image + noise mask),
    # this value and the `unet_input_img` preparation in `main()` must be changed accordingly.
    defaults['in_channels'] = 7

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # Add default arguments to the parser
    return parser

if __name__ == "__main__":
    main()
