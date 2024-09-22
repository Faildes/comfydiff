from .sdxl_clip import SDXLClipModel, SDXLTokenizer

class SDXL:
    def __init__(self, device="cpu", dtype=None,model_options={}):
        model_options["dtype"] = dtype
        self.tokenizer = SDXLTokenizer()
        self.clip = SDXLClipModel(device, dtype)

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.clip.reset_clip_options()

        self.load_model()
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

        cond_len = cond.shape[1]
        uncond_len = uncond.shape[1]
        if cond_len == uncond_len:
            return [cond, cond_pooled], [uncond, uncond_pooled]
        else:
            if cond_len > uncond_len:
                n = (cond_len - uncond_len) // 77
                return [cond, cond_pooled], [torch.cat([uncond] + [self.encode("")]*n, dim=1), uncode_pooled]
            else:
                n = (uncond_len - cond_len) // 77
                return [torch.cat([cond] + [self.encode("")]*n, dim=1),cond_pooled], [uncond, uncond_pooled]
