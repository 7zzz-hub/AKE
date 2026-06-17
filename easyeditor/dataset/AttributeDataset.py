import os
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoProcessor
from collections import defaultdict
import json


class AttributeDataset:
    def __init__(self, data_path, size, config, eval_mode=True):
        self.data_path = data_path
        self.config = config
        self.size = size
   
        tok_name = config.tokenizer_name or config.name
        if config.tokenizer_class == "QWenTokenizer":
            self.tok = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            self.processor = AutoProcessor.from_pretrained(config.name,)
        else:
            self.tok = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )  

        if not self.tok.pad_token:
            self.tok.pad_token = self.tok.eos_token
        
        # prompt
        if config.model_class == "LLaVA":
            self.prompt = "USER: <image>\n{} Please answer in one word. ASSISTANT:"
        elif config.model_class == "Blip2OPT":
            self.prompt = "Question: {} Please answer in one word. Short answer:"
        elif config.model_class == "qwen-vl":
            self.prompt = "{} Please answer in one word."

        self.samples = self.build_data()


    def add_qa_pair(self, sample, key, image, question, answer):
        if key not in sample:
            sample[key] = []

        if self.config.model_class == "blip2":
            answer = " " + answer
        elif self.config.model_class == "LLaVA":
            answer = answer.capitalize()
            
        sample[key].append({
            'image': image,
            'question': question,
            'answer':  answer
        })


    def build_data(self):
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.size is None:
            self.size = len(data)
            
        samples = []
        for record in tqdm(data[:self.size]):

            if record["alt"] == "":
                continue
            
            # all image
            image_path = os.path.join(self.config.image, record["image"])
            re_image = os.path.join(self.config.image, record["rephrase_image"])
            loc_image = os.path.join(self.config.image, record["loc_img"])
            gen_image = os.path.join(self.config.image, record["gen_img"]) if record.get("gen_img") is not None else None
            
            sample = {}

            # rel
            self.add_qa_pair(sample, "rel", image_path, record["src"], record["alt"])

            # rephrase
            self.add_qa_pair(sample, "re_image", re_image, record["src"], record["alt"])

            # loc
            self.add_qa_pair(sample, "Loc_in", loc_image, record["loc_in_q"], "_")
            self.add_qa_pair(sample, "Loc_out", loc_image, record["loc_out_q"], "_")

            # gen
            if gen_image is not None:
                self.add_qa_pair(sample, "gen1", gen_image, record["gen1_q"], record["gen1_a"])
                self.add_qa_pair(sample, "gen2", gen_image, record["gen2_q_1"], record["gen2_a"])
                self.add_qa_pair(sample, "gen2", gen_image, record["gen2_q_2"], record["gen2_a"])

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def build_task(self, images, prompts, targets):

        def qwenvl_qa(image, question):
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1280*28*28,},
                    {"type": "text", "text": self.prompt.format(question)}
                ]
            }]
            
        def concat_qa(image, question, answer):
            if self.config.model_class == "LLaVA":
                return self.prompt.format(question) + " " + answer.capitalize()
            elif self.config.model_class == "Blip2OPT":
                return self.prompt.format(question) + answer
            elif self.config.model_class == "qwen-vl":
                return self.processor.apply_chat_template(qwenvl_qa(image, question), tokenize=False) + " " + answer
        

        text_inputs = [concat_qa(image,p,t) for image, p, t in zip(images, prompts, targets)]
        prompts_len = [len(self.tok.encode(self.prompt.format(p), add_special_tokens=False)) for p in prompts] 
        labels = self.tok(targets, return_tensors="pt", add_special_tokens=False)["input_ids"]
            
        return {
            "image": images,
            "prompts": prompts,
            "text_input": text_inputs,
            "inputs": None,
            "prompts_len": prompts_len if not self.config.model_class=="qwen-vl" else None, 
            "labels": labels
        }

    # =========================
    # collate
    # =========================
    def collate_fn(self, batch):

        keys_name = list(batch[0].keys())
        build_batch = {key: [] for key in keys_name}
        for batch_name, key_name in zip(build_batch.keys(), keys_name):
            for b in batch:
                build_batch[batch_name].append(self.build_task(
                    [e["image"] for e in b[key_name]],
                    [e["question"] for e in b[key_name]],
                    [e["answer"] for e in b[key_name]],
                ))

        return build_batch