from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from .models.modeling_crello import CrelloModel, CrelloModelConfig
import os
from .training.datasets.quantizer import get_quantizer
import click
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from training.utils import accuracy, load_gallery, h_cat, all_gather
from .training.trainer_crello import batch_purity
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import copy
from tqdm import tqdm
import torch.nn.functional as F
import re
# build model and tokenizer
def buildmodel(**kwargs):
    # seed / input model / resume
    resume = kwargs.get('resume', None)
    seed = kwargs.get('seed', None)
    input_model = kwargs.get('input_model', None)
    quantizer_version = kwargs.get('quantizer_version', 'v4')
    
    set_seed(seed)
    old_tokenizer = AutoTokenizer.from_pretrained(input_model, trust_remote_code=True)
    old_vocab_size = len(old_tokenizer)
    # old_vocab_size = 128256
    print(f"Old vocab size: {old_vocab_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(resume, trust_remote_code=True)
   
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")
    quantizer = get_quantizer(quantizer_version, 
                    update_vocab = False, 
                    simplify_json = False, # 简化json
                    num_mask_tokens = 0, # mask token
                    mask_type = kwargs.get('mask_type'), # mask type
                    decimal_quantize_types = kwargs.get('decimal_quantize_types'), # 十进制来表示数字
                    mask_values = kwargs['mask_values'],
                    width = kwargs['width'],
                    height = kwargs['height']
                    )
    quantizer.setup_tokenizer(tokenizer)    
    print(f"latest tokenzier size: {len(tokenizer)}")

    model_args = CrelloModelConfig(
        old_vocab_size = old_vocab_size,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ignore_ids=tokenizer.convert_tokens_to_ids(quantizer.ignore_tokens),
    )

    model_args.opt_version = input_model
    model_args.freeze_lm = False
    model_args.load_in_4bit = kwargs.get('load_in_4bit', False)
    model_args.use_lora = False

    print(f"Resuming from checkpoint {resume}, Waiting to ready")
    model = CrelloModel.from_pretrained(resume, config=model_args)
    tokenizer.add_special_tokens({"mask_token": "<mask>"}) 
    mask_token = tokenizer.mask_token
    quantizer.additional_special_tokens.add("<mask>")
    added_special_tokens_list = ["<layout>", "<position>", "<wholecaption>"] 
    tokenizer.add_special_tokens({"additional_special_tokens": added_special_tokens_list}, replace_additional_special_tokens=False)
    for token in added_special_tokens_list:
        quantizer.additional_special_tokens.add(token)
        
    return model, quantizer, tokenizer


# build data
def FormulateInput(intension: str):
    '''
    Formulate user input string to Dict Object
    '''

    # resdict = {}
    # resdict["intension"] = intension
    # resdict["wholecaption"] = ""
    # resdict["layout"] = []
    
    resdict = {}
    resdict["wholecaption"] = intension
    resdict["layout"] = []
    
    return resdict

# build output
@torch.no_grad()
def generate(model, tokenizer, embeddings = torch.FloatTensor, 
                max_len: int = 1024, temperature: float = 0.0, 
                top_p: float = 1.0, filter_value: float = -float('Inf'),
                invalid_ids: List[int] = [], end_token: str = None, final_tokens: str = None
                ):
    """Runs greedy decoding and returns generated captions.

    Args:
        min_word_tokens: Minimum number of words to generate before allowing a [IMG] output.
        filter_value: Value to assign to tokens that should never be generated.
    Outputs:
        out: (N, T) int32 sequence of output tokens.
        output_embeddings: (N, T, 256) sequence of text output embeddings.
    """
    out_tokens = []
    # init output with image tokens
    for i in range(max_len):
        output = model.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
        logits = output.logits[:, -1, :]  # (N, vocab_size)
        if top_p == 1.0:
            logits = logits.cpu()
        # Prevent the model from generating invalid tokens.
        logits[:, invalid_ids] = filter_value

        if temperature == 0.0:
            if top_p != 1.0:
                raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
            next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
        else:
            logits = logits / temperature
            # Apply top-p filtering.
            if top_p < 1.0:
                assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (N, D)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for j in range(sorted_indices.shape[0]):
                    indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                    logits[j, indices_to_remove] = filter_value

            token_weights = logits.exp()   # (N, vocab_size)
            next_token = torch.multinomial(token_weights, 1)  # (N, 1)
            next_token = next_token.long().to(embeddings.device)
        next_word = tokenizer.convert_ids_to_tokens(int(next_token))
        # print(next_word)
        if end_token in next_word and i!=0:
            break
        out_tokens.append(next_token)
        
        if final_tokens is not None:
            out_text = tokenizer.batch_decode([torch.cat(out_tokens, dim=1).reshape(-1).tolist()])[0]
            if final_tokens in out_text:
                # break
                out_text = out_text.split(final_tokens)[0]
                return tokenizer(out_text, return_tensors="pt").input_ids.cuda()[0]

        next_embedding = model.input_embeddings(next_token.cuda())
        embeddings = torch.cat([embeddings, next_embedding], dim=1)
        
        
    if len(out_tokens) == 0:
        empty_tensor = torch.empty(0, embeddings.shape[-1])  
        return empty_tensor 
    return torch.cat(out_tokens, dim=1)


@torch.no_grad()    
def evaluate_v1(inputs, model, quantizer, tokenizer, width, height):
    json_example = inputs
    ## input_intension = '{"intension":"' + json_example["intension"] + '","wholecaption":"'
    input_intension = '{"wholecaption":"' + json_example["wholecaption"] + '","layout":[{"layer":'
    print("input_intension:\n", input_intension)
    
    inputs = tokenizer(
            input_intension, return_tensors="pt"
        ).to("cuda")
    # outputs = model.lm.generate(**inputs, use_cache=True, max_new_tokens=800)
    outputs = model.lm.generate(**inputs, use_cache=True, max_length=3000)
    
    #---------------------------------------------------------------------
    inputs_length = inputs['input_ids'].shape[1] 
    outputs = outputs[:, inputs_length:]
    #---------------------------------------------------------------------
    
    outputs_word = tokenizer.batch_decode(outputs)[0]
    # print(outputs_word)
    split_word = outputs_word.split('}]}')[0]+"}]}"
    # print(split_word)
    
    #---------------------------------------------------------------------
    # split_word = split_word.replace('\n', '\\n').replace('\\"', '\\\"').replace(' "', ' \\"').replace('" ', '\\" ').replace('\\\\\" ', '\\" ').replace('".', '\\".').replace('", ', '\\", ')
    ## split_word = split_word.replace('\n', '\\n').replace('\\"', '\\\"').replace(' "', ' \\"').replace('" ', '\\" ').replace('\\\\\" ', '\\" ')
    ## split_word = '{"intension":"' + json_example["intension"].replace('"', '\\"') + '","wholecaption":"' + split_word
    split_word = '{"wholecaption":"' + json_example["wholecaption"].replace('\n', '\\n').replace('"', '\\"') + '","layout":[{"layer":' + split_word
    try:
        print("split_word:\n", split_word)
    except:
        print("split_word:\n", split_word.encode('utf-8'))
    #---------------------------------------------------------------------
    map_dict = quantizer.construct_map_dict()
    for key ,value in map_dict.items():
        split_word = split_word.replace(key, value)
    ## split_word = re.sub(r'(<font-\d+>)', r'"\1"', split_word)
    ## split_word = re.sub(r'(<color-\d+>)', r'"\1"', split_word)
    # replaced_tokens = set(self.tokenizer.additional_special_tokens)-set(['<split-text>'])
    # for token in replaced_tokens:
    #     split_word = split_word.replace(token, f'"{token}"')
    try:
        print("split_word:\n", split_word)
    except:
        print("split_word:\n", split_word.encode('utf-8'))
    try:
        pred_json_example = json.loads(split_word)
        for layer in pred_json_example["layout"]:
            layer['x'] = round(int(width)*layer['x'])
            layer['y'] = round(int(height)*layer['y'])
            layer['width'] = round(int(width)*layer['width'])
            layer['height'] = round(int(height)*layer['height'])
    except Exception as e:
        print(e)
        pred_json_example = None
    print("pred_json_example:\n", pred_json_example)
    return pred_json_example

@torch.no_grad()    
def evaluate_v3(inputs, model, quantizer, tokenizer, width, height):
    
    json_example_copy = copy.deepcopy(inputs)
    
    # json_example = quantizer.load_json(json_example_str)
    replaced_token = tokenizer.mask_token
    layout_token = "<layout>"
    position_token = "<position>"

    replaced_id = tokenizer.convert_tokens_to_ids(replaced_token)
    whole_caption_id = tokenizer.convert_tokens_to_ids('<wholecaption>')
    layout_token_id = tokenizer.convert_tokens_to_ids(layout_token)
    position_token_id = tokenizer.convert_tokens_to_ids(position_token)

    final_tokens = '","layout'
    
    field_end_token_dict = {
        'whole_caption': '}',
        'others': '"'
    }

    json_example_copy["wholecaption"] = "<wholecaption>"

    json_example_copy["layout"] = [layout_token]
    
    invalid_ids = []
    new_content = json.dumps(json_example_copy, separators=(',',':'))
    new_content = new_content.replace('"<layout>"', '<layout>')
    print("new_content:\n", new_content)

    acc_input_ids = tokenizer(
        new_content, return_tensors="pt"
    ).input_ids.cuda()[0]
    print("acc_input_ids:\n", tokenizer.batch_decode(acc_input_ids.unsqueeze(0))[0])
    
    count = 0
    prev_id_list = []
    gen_texts = []
    for i, input_id in enumerate(acc_input_ids):
        if input_id == whole_caption_id and i != 0:
            prev_ids = torch.LongTensor(prev_id_list).cuda()
            prev_ids = prev_ids[None,]
            input_embs = model.input_embeddings(prev_ids)
            gen_ids = generate(model, tokenizer, input_embs, invalid_ids=invalid_ids, end_token=field_end_token_dict['whole_caption'], max_len=1024, final_tokens=final_tokens)
            gen_text = tokenizer.batch_decode([gen_ids.reshape(-1).tolist()])[0]
            # print("gen_text: \n", gen_text)
            gen_texts.append(gen_text)
            prev_id_list.extend(gen_ids.reshape(-1).tolist())
            count+=1
        elif input_id == layout_token_id and i != 0:
            stop_flag = False
            layout_list = []
            while not stop_flag:
                layer_input = '{"layer":<mask>,"category":"<mask>","caption":"<mask>"'
                max_len_list_1 = [10, 10, 300]
                gen_layers = []
                layer_input_ids = tokenizer(layer_input, return_tensors="pt").input_ids.cuda()[0]
                max_len_count_1 = 0
                for j, layer_id in enumerate(layer_input_ids):
                    if layer_id == replaced_id and j != 0 :
                        prev_ids = torch.LongTensor(prev_id_list).cuda()
                        prev_ids = prev_ids[None,]
                        input_embs = model.input_embeddings(prev_ids)
                        gen_ids = generate(model, tokenizer, input_embs, invalid_ids=invalid_ids, end_token=field_end_token_dict['others'], max_len=max_len_list_1[max_len_count_1])
                        gen_text = tokenizer.batch_decode([gen_ids.reshape(-1).tolist()])[0]
                        # print("gen_text: \n", gen_text)
                        gen_layers.append(gen_text)
                        prev_id_list.extend(gen_ids.reshape(-1).tolist())
                        max_len_count_1 += 1
                    else:
                        prev_id_list.append(layer_id)
                print("gen_layers: \n", gen_layers)
                text_flag = False
                if 'text' in gen_layers[1]:
                    layer_input = ',"font":<mask>,"color":<mask>,"x":<mask>,"y":<mask>,"width":<mask>,"height":<position>}'
                    text_flag = True
                    max_len_list_2 = [10, 10, 10, 10, 10, 20]
                else:
                    layer_input = ',"x":<mask>,"y":<mask>,"width":<mask>,"height":<position>}'
                    max_len_list_2 = [10, 10, 10, 20]
                layer_input_ids = tokenizer(layer_input, return_tensors="pt").input_ids.cuda()[0]
                max_len_count_2 = 0
                for j, layer_id in enumerate(layer_input_ids):
                    if (layer_id == replaced_id or layer_id == position_token_id) and j != 0 :
                        prev_ids = torch.LongTensor(prev_id_list).cuda()
                        prev_ids = prev_ids[None,]
                        input_embs = model.input_embeddings(prev_ids)
                        gen_ids = generate(model, tokenizer, input_embs, invalid_ids=invalid_ids, end_token=field_end_token_dict['others'], max_len=max_len_list_2[max_len_count_2])
                        gen_text = tokenizer.batch_decode([gen_ids.reshape(-1).tolist()])[0]
                        
                        if layer_id == position_token_id:
                            new_gen_ids = generate(model, tokenizer, input_embs, invalid_ids=invalid_ids, end_token=';', max_len=max_len_list_2[max_len_count_2])
                            new_gen_text = tokenizer.batch_decode([new_gen_ids.reshape(-1).tolist()])[0]
                            # print("gen_text: \n", gen_text)
                            # print("new_gen_text: \n", new_gen_text)
                            if new_gen_text[len(gen_text):][:2] == '},':
                                pass
                            else:
                                stop_flag = True
                            gen_text = '<' + gen_text.split('>')[0].split('<')[1] + '>'
                        gen_layers.append(gen_text)
                        prev_id_list.extend(gen_ids.reshape(-1).tolist())
                        max_len_count_2 += 1
                    else:
                        prev_id_list.append(layer_id)
                if text_flag:
                    layout_list.append(
                        {
                            "layer": gen_layers[0],
                            "category": gen_layers[1],
                            "caption": gen_layers[2],
                            "font": gen_layers[3],
                            "color": gen_layers[4],
                            "x": gen_layers[5],
                            "y": gen_layers[6],
                            "width": gen_layers[7],
                            "height": gen_layers[8]
                        }
                    )
                else:
                    layout_list.append(
                        {
                            "layer": gen_layers[0],
                            "category": gen_layers[1],
                            "caption": gen_layers[2],
                            "x": gen_layers[3],
                            "y": gen_layers[4],
                            "width": gen_layers[5],
                            "height": gen_layers[6]
                        }
                    )
                
                prev_id_list.append(tokenizer.convert_tokens_to_ids(','))
            
        else:
            prev_id_list.append(input_id)
    pred_ids = torch.LongTensor(prev_id_list)
    pred_content = batch_purity(pred_ids[None], tokenizer)[0]
    print("pred_content: \n", pred_content)
    
    json_example_copy["wholecaption"] = gen_texts[0]
    json_example_copy["layout"] = layout_list

    print("pred_json_example:\n", json_example_copy)
    map_dict = quantizer.construct_map_dict()
    try:
        for layer in json_example_copy["layout"]:
            layer['x'] = round(int(width)*float(map_dict[layer['x']]))
            layer['y'] = round(int(height)*float(map_dict[layer['y']]))
            layer['width'] = round(int(width)*float(map_dict[layer['width']]))
            layer['height'] = round(int(height)*float(map_dict[layer['height']]))
    except:
        json_example_copy = None
    print("pred_json_example:\n", json_example_copy)
    return json_example_copy

def construction_layout():
    params_dict = {  
        "input_model": "/openseg_blob/v-sirui/temporary/2024-02-21/Layout_train/COLEv2/Design_LLM/checkpoint/Meta-Llama-3-8B",  
        "resume": "/openseg_blob/v-sirui/temporary/2024-02-21/SVD/Int2lay_1016/checkpoint/int2lay_1016/1017_test/tmp-checkpoint-26000",  
        "seed": 0,  
        "mask_values": False,  
        "quantizer_version": 'v4',  
        "mask_type": 'cm3',  
        "decimal_quantize_types": [],  
        "num_mask_tokens": 0,  
        "width": 512,
        "height": 512,
    } 
    # Init model
    model, quantizer, tokenizer= buildmodel(**params_dict)
    model = model.cuda()
    model = model.bfloat16()
    model.eval()
    
    return model, quantizer, tokenizer, params_dict["width"], params_dict["height"]
    
def inference(generate_method, intention, model, quantizer, tokenizer, width, height):
    json_file = "/home/wyb/openseg_blob/v-yanbin/GradioDemo/LLM-For-Layout-Planning/inference_test_sirui.json"
    rawdata = FormulateInput(intention)
    
    if generate_method == 'v1':
        preddata = evaluate_v1(rawdata, model, quantizer, tokenizer, width, height)
    elif generate_method == 'v3':
        preddata = evaluate_v3(rawdata, model, quantizer, tokenizer, width, height)
    else:
        print("Please input correct generate method")
        exit(0)
        
    print(preddata)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as file:
        json.dump(preddata, file, indent=4, ensure_ascii=False)
        
    return preddata, json_file
    
if __name__ == '__main__':
    test_dict = {
        "intention": "Design an engaging and vibrant recruitment advertisement for our company. The image should feature three animated characters in a modern cityscape, depicting a dynamic and collaborative work environment. Incorporate a light bulb graphic with a question mark, symbolizing innovation, creativity, and problem-solving. Use bold text to announce \"WE ARE RECRUITING\" and provide the company's social media handle \"@reallygreatsite\" and a contact phone number \"+123-456-7890\" for interested individuals. The overall design should be playful and youthful, attracting potential recruits who are innovative and eager to contribute to a lively team.",
        "save_path": "/home/wyb/openseg_blob/v-yanbin/GradioDemo/LLM-For-Layout-Planning/inference_test_sirui.json",
        "generate_method": "v1"
    }

    model, quantizer, tokenizer, width, height = construction_layout()
    
    
    preddata, json_file = inference(test_dict["generate_method"], test_dict["intention"], model, quantizer, tokenizer, width, height)
    print(f'preddata: !!!!! \n {preddata}')
    # with open(test_dict['save_path'], 'w') as file:
    #     print(test_dict['save_path'])
    #     json.dump(preddata, file, indent=4, ensure_ascii=False)
