from . import sdxl_clip
import torch

class SDXL:
    def __init__(self, device="cpu", dtype=None):
        self.tokenizer = sdxl_clip.SDXLTokenizer()
        self.clip = sdxl_clip.SDXLClipModel(device, dtype)

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.clip.reset_clip_options()
        o = self.clip.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        cond, pooled = self.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooled
    
    def encode_equal_len(self, prompt, negative_prompt):
        cond, cond_pooled = self.encode(prompt)
        uncond, uncond_pooled = self.encode(negative_prompt)

        empty = self.encode("")[0]
        cond_shape = cond.shape[0]
        empty_z = torch.cat([empty] * cond_shape)
        max_token_count = max([cond.shape[1], uncond.shape[1]])
        while cond.shape[1] < max_token_count:
            cond = torch.cat([cond, empty_z], dim=1)
        while uncond.shape[1] < max_token_count:
            uncond = torch.cat([uncond, empty_z], dim=1)
        return cond, cond_pooled, uncond, uncond_pooled
