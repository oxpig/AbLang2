import os, subprocess, json, argparse,requests
import torch

list_of_models = {
    "ablang1-heavy":["https://opig.stats.ox.ac.uk/data/downloads/ablang-heavy.tar.gz", "amodel.pt"], 
    "ablang1-light":["https://opig.stats.ox.ac.uk/data/downloads/ablang-light.tar.gz", "amodel.pt"],
    "ablang2-paired":["https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz", "model.pt"],
    "tcrlang-paired":["https://zenodo.org/records/11208211/files/tcrlang-weights.tar.gz", "model.pt"],
}
ablang1_models = ["ablang1-heavy", "ablang1-light"]
ablang2_models = ["ablang2-paired", "tcrlang-paired"]


def load_model(model_to_use = "ablang2-paired", random_init = False, device = 'cpu'):

    if model_to_use in ablang1_models:
        AbLang, tokenizer, hparams = fetch_ablang1(
            model_to_use, 
            random_init=random_init, 
            device=device
        )
    elif model_to_use in ablang2_models:
        AbLang, tokenizer, hparams = fetch_ablang2(
            model_to_use, 
            random_init=random_init, 
            device=device
        )
    elif "ABLANG-" in model_to_use:
        AbLang, tokenizer, hparams = fetch_ablang2(
            model_to_use, 
            random_init=random_init, 
            device=device
        )
    else: 
        assert False, f"The selected model to use ({model_to_use}) does not exist.\
        Please select a valid model."   

    return AbLang, tokenizer, hparams
    
    
def download_model(model_to_use = "ablang2-paired"):
    """
    If not already downloaded, download the model.

    Weights live in `model-weights-<model>/` inside the package, unless the
    `ABLANG2_WEIGHTS_DIR` environment variable names another directory.
    """

    weights_dir = os.environ.get("ABLANG2_WEIGHTS_DIR") or os.path.dirname(__file__)
    local_model_folder = os.path.join(weights_dir, "model-weights-{}".format(model_to_use))
    os.makedirs(local_model_folder, exist_ok = True)

    file_w_weights, file_model = list_of_models[model_to_use] # modify list of models

    if not os.path.isfile(os.path.join(local_model_folder, file_model)):
        print("Downloading model ...")
        tmp_file = os.path.join(local_model_folder, "tmp.tar.gz")

        with open(tmp_file,'wb') as f: f.write(requests.get(file_w_weights).content)

        subprocess.run(["tar", "-zxvf", tmp_file, "-C", local_model_folder], check = True) 
        os.remove(tmp_file)
    
    return local_model_folder
    
        
def fetch_ablang1(model_to_use, random_init=False, device='cpu'):
    
    from .models.ablang1 import model as ablang_1_model
    from .models.ablang1 import tokenizers as ablang_1_tokenizer
    
    local_model_folder = download_model(model_to_use)
        
    with open(os.path.join(local_model_folder, 'hparams.json'), 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    

    AbLang = ablang_1_model.AbLang(hparams)
    if not random_init:
        AbLang.load_state_dict(
            torch.load(
                os.path.join(local_model_folder, 'amodel.pt'),
                map_location=torch.device(device)
            )
        )
    tokenizer = ablang_1_tokenizer.ABtokenizer(os.path.join(local_model_folder, 'vocab.json'))
        
    return AbLang, tokenizer, hparams


def fetch_ablang2(model_to_use, random_init=False, device='cpu'):
    
    from .models.ablang2 import ablang
    from .models.ablang2 import tokenizers
    
    if model_to_use in ablang2_models:
        local_model_folder = download_model(model_to_use)
    else:
        local_model_folder = model_to_use
    
    with open(os.path.join(local_model_folder, 'hparams.json'), 'r', encoding='utf-8') as f:
        hparams = argparse.Namespace(**json.load(f))    
        
    AbLang = ablang.AbLang(
        vocab_size = hparams.vocab_size,
        hidden_embed_size = hparams.hidden_embed_size,
        n_attn_heads = hparams.n_attn_heads,
        n_encoder_blocks = hparams.n_encoder_blocks,
        padding_tkn = hparams.pad_tkn,
        mask_tkn = hparams.mask_tkn,
        layer_norm_eps = hparams.layer_norm_eps,
        a_fn = hparams.a_fn,
    )

    if not random_init:
        AbLang.load_state_dict(
            torch.load(
                os.path.join(local_model_folder, 'model.pt'), 
                map_location=torch.device(device)
            )
        )
    tokenizer = tokenizers.ABtokenizer()
    
    return AbLang, tokenizer, hparams
