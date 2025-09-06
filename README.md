<h1 align="center">🎇AKEdit (Attribute Knowledge Editing)</h1>


### 📥Dataset: CLEVR
CLEVR is a diagnostic dataset designed to evaluate **compositional language understanding** and **visual reasoning** in AI models.

**Dataset Details:**  
- **Official Paper:** [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://cs.stanford.edu/people/jcjohns/clevr/)  
- **Download Link:** [CLEVR_CoGenT_v1.0.zip](https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip)  

---

### 🤖Supported Models & Methods

#### Models (`--model`)
| Model  | Description                  |
|--------|------------------------------|
| blip2  | BLIP-2 multimodal model      |
| llava  | LLaVA vision-language model  |

#### Editing Methods (`--method`)
| Method | Description                              |
|--------|------------------------------------------|
| FT-L   | Fine-tuning on language (last layer)     |
| FT-V   | Fine-tuning on vision (Q-Former)        |
| MEND   |   |
| SERAC  |  |
| (mllm edit)   |   |
| Ours  |  |

---

### 🚀How to Run

1. **Enter Project Directory**
```bash
cd AKE-main
```
2. **install Dependencies**
<br> Please refer to the envs/vlkeb_easyedit.yml file for environment setup.

3. **Execute Editing Task**
Basic command format:
```bash
python multimodal_edit.py \
    --model <MODEL_NAME> \
    --method <EDITING_METHOD> \
    --size <DATASET_SIZE>
```
4. **Example**
```bach
python multimodal_edit.py --model blip2 --method FT-L --size 1000
```

---

### 🙂Pre-trained Model

To run the code, we also need to download the pre-trained pytorch models of LVLMs and others, then put them in proper directories.

Here we put under 'huggingface_cache' folder:
```bash
# models in hugging_cache folder
huggingface_cache/
├── bert-base-uncased/
├── distilbert-base-cased/
├── llava-v1.5-7b/
├── opt-6.7b/
├── opt-1.3b/
├── clip-vit-large-patch14-336/
│   
├── blip2_pretrained_flant5xxl.pth
├── blip2_pretrained_opt6.7b.pth
└── eva_vit_g.pth
``` 
Links are in the following:
<table>
    <tr>
        <td><a href="https://huggingface.co/google-bert/bert-base-uncased">bert-base-uncased</a></td>
        <td><a href="https://huggingface.co/distilbert/distilbert-base-cased">distilbert-base-cased</a></td>
        <td><a href="https://huggingface.co/liuhaotian/llava-v1.5-7b">llava-v1.5-7b</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/opt-6.7b">opt-6.7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-1.3b">opt-1.3b</a></td>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth">blip2_pretrained_opt6.7b.pth</a></td>
    </tr>
    <tr>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth">eva_vit_g.pth</a></td>
        <td><a href="https://huggingface.co/openai/clip-vit-large-patch14-336">clip-vit-large-patch14-336</a></td>
        <td></td>
    </tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
