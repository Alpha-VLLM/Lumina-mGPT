import json
import logging
import random
from typing import Dict, List

from PIL import Image
import torch

from data.convertsation import Conversation
import model.chameleon_vae_ori as chameleon_vae_ori
from xllmx.data.data_reader import read_general
from xllmx.data.item_processor import MMConvItemProcessor

logger = logging.getLogger(__name__)


def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


class FlexARItemProcessor(MMConvItemProcessor):
    image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
    image_end_token = "<eoss>"
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"

    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
        conv_template=Conversation,
        target_size=512,
    ):

        super().__init__(
            {
                "<|image|>": self.process_image,
            },
            ["<|image|>"],
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        logger.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))

        #  todo
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device="cuda")
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path="./ckpts/chameleon/tokenizer/vqgan.yaml",
            ckpt_path="./ckpts/chameleon/tokenizer/vqgan.ckpt",
            device="cuda",
        )

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @torch.no_grad()
    def process_image(self, image) -> Dict:
        if isinstance(image, Image.Image):
            pass
        else:
            image = Image.open(read_general(image))

        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
        ).view(-1)

        full_image_toks = image_toks.reshape(image.size[1] // 16, image.size[0] // 16)
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size[1] // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]

        return {"input_ids": result_toks, "labels": result_toks}

    def process_item(self, item, training_mode=False, out_flatten=True):
        if not out_flatten:
            return super().process_item(item, training_mode=training_mode)

        if training_mode:
            tokens, labels = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            modified_labels_item = []
            for i, (token_or_media, ori_label) in enumerate(zip(tokens, labels)):
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:
                    input_tokens_item += token_or_media["input_ids"]
                    if ori_label <= 0:  # in the prompt part
                        modified_labels_item += [-100] * len(token_or_media["input_ids"])
                    else:
                        modified_labels_item += token_or_media["labels"]

            return input_tokens_item, modified_labels_item
        else:
            tokens = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            for i, token_or_media in enumerate(tokens):
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item

    def decode_image(self, tokens: List[int]) -> Image.Image:
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        h, w = h_grids * self.patch_size, w_grids * self.patch_size
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).cuda()

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten()

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(tokens, h_latent_dim, w_latent_dim)
