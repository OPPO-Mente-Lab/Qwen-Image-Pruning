
import json

import braceexpand
import webdataset as wds
from tqdm import tqdm
import torch
from torchvision.transforms.functional import crop
import re
from torchvision import transforms
import random

import numpy as np
import os,glob
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5EncoderModel,MT5EncoderModel,AutoTokenizer,AutoModel,AutoProcessor
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel

from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import random_split
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
import random

def extract_chinese(text):
    return ''.join(re.findall(r'[\u4e00-\u9fa5]', text))

class InverseNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 反向标准化: (tensor * std) + mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train',
                            default=False, action="store_true")
        parser.add_argument('--resample_train',
                            default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str,
                            default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)
        parser.add_argument(
            "--resolution", type=int, default=1328,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=False,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer_qwen,
        custom_collate_fn=None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer_qwen = tokenizer_qwen
        self.collate_fn = custom_collate_fn if custom_collate_fn is not None else collate_fn
        self.center_crop = args.center_crop

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None


        self.prepare_data()
        self.setup()

    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1

        all_urls = []
        for url in self.webdataset_base_urls:
            if "*" in url:
                all_urls += expand_urls1(url)
            else:
                all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + \
            num_val == len(
                all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(
            all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)

    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer_qwen,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
            )

            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)

    def _train_dataloader(self):

        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # return DataLoader(
        # num_workers=self.num_workers,
        return DataLoader(
            dataset=self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

TETX_ENCODER = "chinese_clip"  ## mul_clip  chinese_clip  mt5  alt_clip

USED_KEYS = {"1.0.jpg": "instance_images","json": "instance_prompt_ids"}

BUCKETS = [[1328, 1328]]
BUCKET_PROBS = [1.0]

MAX_AR_ERROR = 1.2
ASPECTS = np.array([b[0]/b[1] for b in BUCKETS])


def str_contain_chinese(str):
    for ch in str:
        if u'\u4e00'<=ch<=u'\u9fff':
            return True
    return False


def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)

def expand_urls1(urls):
    result = []
    for file_ in glob.glob(urls):
        result.append(file_)
    return result


def contains_quote(s):
    quotes = {"'", '"', '`', '‘', '’', '“', '”'}
    return bool(set(s) & quotes)  # 有交集即为 True

def verify_keys(samples, required_keys, handler=wds.handlers.warn_and_continue):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """

    for sample in samples:
        try:
            sample_json = sample["json"]
            if "jpg" in sample:
                sample["1.0.jpg"] = sample["jpg"]
            w, h = sample["1.0.jpg"].size
        except:
            print("#######sample",sample)
            continue

        is_normal = True
        aspect = float(w)/float(h)
        if "text_accuray" in sample_json and sample_json["text_accuray"]<1:
            continue

        bucket_id = np.abs(ASPECTS - aspect).argmin()
        if abs(ASPECTS[bucket_id] - aspect) < MAX_AR_ERROR:
            sample["bucket_id"] = bucket_id
        for key in required_keys:
            if key not in sample:
                print(f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}")
                is_normal = False
        if is_normal:
            yield {key: sample[key] for key in required_keys}



key_verifier = wds.filters.pipelinefilter(verify_keys)



class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer_qwen = None,
            extra_keys=["bucket_id"],
            handler=wds.handlers.warn_and_continue,
            resample=False,
            shuffle_shards=True,
            center_crop=False
    ):

        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        self.center_crop = center_crop
        self.crop = transforms.CenterCrop(1328)
        self.resampling = resample
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer_qwen = tokenizer_qwen

        self.append(wds.ResampledShards(urls))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifier(required_keys=keys, handler=handler))
        # Apply preprocessing
        self.append(wds.map(self.preproc))
        # self.append(wds.to_tuple(*keys))

        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128


    def preproc(self, sample):
        """Applies the preprocessing for images"""
        example = {}

        instance_image = sample["1.0.jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)

        sample_json = sample["json"]
        if "prompt_zh" in sample_json:
            example["instance_prompt_ids"] = sample_json["prompt_zh"]
        elif "prompt_en" in sample_json:
            example["instance_prompt_ids"] = sample_json["prompt_en"]
        elif "caption_en" in sample_json:
            example["instance_prompt_ids"] = sample_json["caption_en"]  
        else:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            example["instance_prompt_ids"] = ""


        # print(example["instance_prompt_ids"])

        prompt = [example["instance_prompt_ids"]]
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        # txt_tokens = self.tokenizer_qwen(txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt")
        txt_tokens = self.tokenizer_qwen(txt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        example["mllm_input_ids"] = txt_tokens.input_ids
        example["mllm_attention_mask"] = txt_tokens.attention_mask

        return example


def collate_fn(examples):
    mllm_input_ids = [example["mllm_input_ids"] for example in examples]
    mllm_attention_mask = [example["mllm_attention_mask"] for example in examples]


    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # texts_en = [example["instance_en"] for example in examples]

    batch = {
        "input_ids": torch.cat(mllm_input_ids),
        "attention_mask": torch.cat(mllm_attention_mask),
        "pixel_values": pixel_values,
    }


    return batch
