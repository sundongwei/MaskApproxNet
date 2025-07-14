import sys
import argparse
import json # Added for loading vocab
import os # Added for path joining for vocab
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
# from data.whu.Whu_CC.whucd import whuDataset # Old dataset
from data.whucc.Whu_CC.whucc import WHUCCDataset # New dataset
from module.decoder import DecoderTransformerWithMask # Added for caption decoder
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom # Visdom is commented out, so keeping it as is.
# viz = Visdom(port=8850)
# import torchvision.transforms as transforms # Not directly used in this script, but WHUCCDataset uses it.

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'whu': # Keep this check if other datasets might be added later,
                                # otherwise, it can be simplified.
        print("Using WHUCCDataset (modified for captions and change masks)")
        ds = WHUCCDataset(
            data_folder=args.data_dir, # This is args.data_dir from the original script
            list_path=args.list_path_dir,
            split='train',
            token_folder=args.token_dir,
            vocab_file=args.vocab_name,
            max_length=args.max_caption_length,
            allow_unk=args.allow_unk
        )
    else:
        raise ValueError(f"Unsupported dataset name: {args.data_name}")

    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )
    # data = iter(datal) # Keep this for TrainLoop, but for verification, we'll iterate once.

    # Verification step: print shapes of the first batch
    logger.log("Verifying data loader output...")
    try:
        first_batch = next(iter(datal))
        imgA, imgB, label, token_all, token_all_len, token, token_len, name = first_batch
        logger.log(f"  imgA shape: {imgA.shape}")
        logger.log(f"  imgB shape: {imgB.shape}")
        logger.log(f"  label shape: {label.shape}")
        logger.log(f"  token_all shape: {token_all.shape}")
        logger.log(f"  token_all_len shape: {token_all_len.shape}")
        logger.log(f"  token shape: {token.shape}")
        logger.log(f"  token_len shape: {token_len.shape}")
        logger.log(f"  name (first item): {name[0]}")
        logger.log(f"  Label dtype: {label.dtype}, Label min: {label.min()}, Label max: {label.max()}")

    except Exception as e:
        logger.log(f"Error during data loader verification: {e}")
        import traceback
        traceback.print_exc()
        # Exit if dataloader verification fails, as training would also fail.
        # Depending on the setup, you might want to sys.exit(1) here.
        # For now, just log and continue to model creation to see if other parts are okay.

    data = iter(datal) # Re-assign for TrainLoop

    data = iter(datal) # Re-assign for TrainLoop

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)

    logger.log("loading vocabulary and creating caption decoder...")
    vocab_path = os.path.join(args.list_path_dir, args.vocab_name + '.json')
    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f)
    vocab_size = len(word_vocab)

    caption_decoder = DecoderTransformerWithMask(
        encoder_dim=1,  # The change map from diffusion model will have 1 channel
        feature_dim=args.decoder_feature_dim,
        vocab_size=vocab_size,
        max_lengths=args.max_caption_length,
        word_vocab=word_vocab,
        n_head=args.decoder_n_head,
        n_layers=args.decoder_n_layers,
        dropout=args.decoder_dropout
    )

    if args.multi_gpu:
        caption_decoder = th.nn.DataParallel(caption_decoder, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        caption_decoder.to(device=th.device('cuda', int(args.gpu_dev)))
    else:
        caption_decoder.to(dist_util.dev())

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None, # Keep this as None, not used
        caption_decoder=caption_decoder, # Pass the caption decoder
        data=data, # iterator from dataloader
        dataloader=datal, # dataloader itself (for length, etc.)
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'whu', # Keep this as it might be used for other logic, though dataset class is now WHUCCDataset
        data_dir='./data/whu/Whu_CC/whu_CDC_dataset/images', # Corrected to point to 'images' subdirectory as per WHUCCDataset's expectation
        list_path_dir='./data/whu/Whu_CC/', # Path to train.txt, val.txt, vocab.json
        token_dir='./data/whu/Whu_CC/tokens/', # Path to token files
        vocab_name='vocab', # Basename for vocab.json
        max_caption_length=40, # Used by WHUCCDataset and DecoderTransformerWithMask
        allow_unk=1, # Allow unknown tokens by WHUCCDataset
        # Decoder Hyperparameters
        decoder_feature_dim=512,
        decoder_n_head=8,
        decoder_n_layers=6,
        decoder_dropout=0.1,
        # Training Hyperparameters
        caption_loss_lambda=0.1, # Verified: Default weight for the captioning loss.
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results_whu_without_freq/'
    )
    model_diff_defaults = model_and_diffusion_defaults()
    model_diff_defaults['in_channels'] = 4 # Override in_channels to 4
    defaults.update(model_diff_defaults)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
