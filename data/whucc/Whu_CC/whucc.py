import json
import os
import sys
sys.path.append("/root/siton-data-114470f9d4a4445abe441748148716da/Chg2Cap-main/")
import numpy as np
import torch
import torch.utils.data as data
from imageio import imread
from preprocess_data import encode
from random import *
from PIL import Image
import torchvision.transforms as transforms

class WHUCCDataset(data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 40, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored (e.g., './data/whu/Whu_CC/whu_CDC_dataset/images')
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored (e.g., './data/whu/Whu_CC/')
        :param split: split, one of 'train', 'val', or 'test'
        :param token_folder: folder where token files are stored (e.g., './data/whu/Whu_CC/tokens/')
        :param vocab_file: the name of vocab file (e.g., 'vocab')
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.data_folder = data_folder
        self.list_path = list_path
        self.split = split
        self.token_folder = token_folder
        self.vocab_file = vocab_file
        self.max_length = max_length
        self.allow_unk = allow_unk

        self.mean=[123.8806, 118.6361, 108.5552]
        self.std=[49.5580, 47.7908, 50.2587]

        assert self.split in {'train', 'val', 'test'}

        self.imageA_dir = os.path.join(self.data_folder, self.split, 'A')
        self.imageB_dir = os.path.join(self.data_folder, self.split, 'B')
        self.label_dir = os.path.join(self.data_folder, self.split, 'label')

        self.img_ids = [i_id.strip() for i_id in open(os.path.join(self.list_path, self.split + '.txt'))]

        if self.vocab_file is not None:
            with open(os.path.join(self.list_path, self.vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)

        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        self.files = []
        for name_entry in self.img_ids:
            base_name = name_entry.split('-')[0] # e.g. test_000024.tif from test_000024.tif-0 or train_00001.tif from train_00001.tif-0

            img_fileA = os.path.join(self.imageA_dir, base_name)
            img_fileB = os.path.join(self.imageB_dir, base_name)
            label_file = os.path.join(self.label_dir, base_name)

            token_id = None
            token_file = None

            if self.split == 'train':
                token_id = name_entry.split('-')[-1] # This is the caption index
                if self.token_folder is not None:
                    # token filename is like train_00001.txt (not train_00001.tif.txt)
                    token_file = os.path.join(self.token_folder, base_name.split('.')[0] + '.txt')
            elif self.split == 'val' or self.split == 'test':
                # For val and test, we might not have a specific token_id from the filename,
                # and tokens might be structured differently or all captions are used.
                if self.token_folder is not None:
                    token_file = os.path.join(self.token_folder, base_name.split('.')[0] + '.txt')

            self.files.append({
                "imgA": img_fileA,
                "imgB": img_fileB,
                "label": label_file,
                "token": token_file,
                "token_id": token_id, # This will be None for val/test if not specified in file list
                "name": base_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = imread(datafiles["imgA"])
        imgB = imread(datafiles["imgB"])
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)   
        imgA = np.moveaxis(imgA, -1, 0)     
        imgB = np.moveaxis(imgB, -1, 0)

        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]

        label = torch.empty(0) # Placeholder for label
        if self.split != 'test': # Load labels for train and val
            if os.path.exists(datafiles["label"]):
                label_pil = Image.open(datafiles["label"]).convert('L')

                transform_list = [
                    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ]
                label_transform = transforms.Compose(transform_list)
                label = label_transform(label_pil)
                label = (label > 0.5).float()
            else: # If label file is missing for some reason
                print(f"Warning: Label file not found for {datafiles['label']}")
                # Return a tensor of expected shape but perhaps zeros, or handle error
                label = torch.zeros(1, 256, 256).float() # Assuming 1 channel, 256x256
        elif self.split == 'test':
             # For test set, if labels are not used/available, return name or placeholder
             # If your test evaluation requires dummy labels of the correct type/device:
             label = name # Or some other indicator if actual label data isn't loaded/processed.
                         # The original whucd.py returned img_name for the test label.
                         # Let's return the name for now, it will be filtered out by the dataloader collate_fn if not a tensor.
                         # Or, more consistently, return an empty tensor.
             label = torch.empty(0)


        if datafiles["token"] is not None and os.path.exists(datafiles["token"]):
            with open(datafiles["token"]) as f:
                caption_text = f.read()
            caption_list = json.loads(caption_text)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens_str in enumerate(caption_list):
                tokens_encode = encode(tokens_str, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)

            if datafiles["token_id"] is not None: # Primarily for 'train' split
                cap_idx = int(datafiles["token_id"])
                token = token_all[cap_idx]
                token_len = token_all_len[cap_idx].item()
            else: # For 'val' or if 'test' has tokens but no specific id (e.g. choose random or use all)
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            # Fallback if token file doesn't exist or not specified
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = 0 # scalar
            token_all_len = np.zeros(1, dtype=int)


        return imgA.copy(), imgB.copy(), label.clone() if torch.is_tensor(label) else label, token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name
    
if __name__ == "__main__":
    data_folder = './data/Whu_CC/whu_CDC_dataset/images'
    list_path = '../../../../data/whu/Whu_CC/'  # Adjusted for relative path if script is run from its directory
    split = 'train'
    # Note: The original main block calculates mean/std. This might need adjustment
    # if the dataloader's return tuple changes significantly or if `label` is not a tensor for 'test'.
    # For now, let's test basic instantiation and item retrieval.

    # Test instantiation
    print(f"Initializing WHUCCDataset for split: {split}")
    # These paths need to be correct for your local setup to run this __main__ block
    # Assuming the script is in Chg2Cap-main/data/whucc/Whu_CC/
    # And data is in Chg2Cap-main/data/whu/Whu_CC/

    # Corrected paths for typical execution from project root or if this script is moved/linked
    # For example, if Chg2Cap-main is the root:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '../../../../')) # Adjust as needed

    # data_folder_abs = os.path.join(project_root, 'data/whu/Whu_CC/whu_CDC_dataset/images')
    # list_path_abs = os.path.join(project_root, 'data/whu/Whu_CC/')
    # token_folder_abs = os.path.join(project_root, 'data/whu/Whu_CC/tokens/')

    # Simpler paths if running from Chg2Cap-main root:
    data_folder_main = './data/whu/Whu_CC/whu_CDC_dataset/images'
    list_path_main = './data/whu/Whu_CC/'
    token_folder_main = './data/whu/Whu_CC/tokens/'
    vocab_file_main = 'vocab'

    print(f"Data folder: {data_folder_main}")
    print(f"List path: {list_path_main}")
    print(f"Token folder: {token_folder_main}")

    try:
        train_dataset = WHUCCDataset(data_folder=data_folder_main,
                                     list_path=list_path_main,
                                     split=split,
                                     token_folder=token_folder_main,
                                     vocab_file=vocab_file_main,
                                     max_length=40,
                                     allow_unk=1)
        print(f"Successfully initialized WHUCCDataset for {split} split.")
        print(f"Number of samples: {len(train_dataset)}")

        if len(train_dataset) > 0:
            print("\nAttempting to get the first item...")
            # Test __getitem__
            imgA, imgB, label, token_all, token_all_len, token, token_len, name = train_dataset[0]
            print(f"Successfully retrieved first item for {split} split: {name}")
            print(f"  imgA shape: {imgA.shape}, dtype: {imgA.dtype}")
            print(f"  imgB shape: {imgB.shape}, dtype: {imgB.dtype}")
            if torch.is_tensor(label):
                print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
            else:
                print(f"  Label: {label} (likely for test split or if loading failed)")
            print(f"  Token (single example): {token[:10]}..., len: {token_len}")
            print(f"  Token_all shape: {token_all.shape}")
            print(f"  Token_all_len shape: {token_all_len.shape}")

        # Test 'val' split
        print("\nInitializing WHUCCDataset for split: val")
        val_dataset = WHUCCDataset(data_folder=data_folder_main,
                                   list_path=list_path_main,
                                   split='val',
                                   token_folder=token_folder_main,
                                   vocab_file=vocab_file_main)
        print(f"Successfully initialized WHUCCDataset for val split.")
        print(f"Number of samples: {len(val_dataset)}")
        if len(val_dataset) > 0:
            print("\nAttempting to get the first item for val split...")
            imgA_val, imgB_val, label_val, _, _, _, _, name_val = val_dataset[0]
            print(f"Successfully retrieved first item for val split: {name_val}")
            if torch.is_tensor(label_val):
                 print(f"  Val Label shape: {label_val.shape}, dtype: {label_val.dtype}")
            else:
                print(f" Val Label: {label_val}")


        # Test 'test' split
        print("\nInitializing WHUCCDataset for split: test")
        test_dataset = WHUCCDataset(data_folder=data_folder_main,
                                   list_path=list_path_main,
                                   split='test',
                                   token_folder=token_folder_main,
                                   vocab_file=vocab_file_main) # Assuming vocab might still be useful for test
        print(f"Successfully initialized WHUCCDataset for test split.")
        print(f"Number of samples: {len(test_dataset)}")

        if len(test_dataset) > 0:
            print("\nAttempting to get the first item for test split...")
            imgA_test, imgB_test, label_test, _, _, _, _, name_test = test_dataset[0]
            print(f"Successfully retrieved first item for test split: {name_test}")
            if torch.is_tensor(label_test):
                print(f"  Test Label shape: {label_test.shape}, dtype: {label_test.dtype}") # Should be empty tensor
            else:
                print(f"  Test Label: {label_test}") # Should be empty tensor or specific placeholder like 'name'

    except Exception as e:
        print(f"Error during WHUCCDataset test: {e}")
        import traceback
        traceback.print_exc()