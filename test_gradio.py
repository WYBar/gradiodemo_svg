import os
import argparse
from tqdm import tqdm
from mmengine.config import Config

import ast
import random
import numpy as np
from functools import partial

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision.utils import save_image

from diffusers import FluxTransformer2DModel
from diffusers.utils import check_min_version
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

import gradio as gr
from PIL import Image
import xml.etree.cElementTree as ET
from io import BytesIO
import xml.etree.cElementTree as ET
import base64
import re

# from .custom_dataset import MultiLayerDataset, general_collate_fn
# from .custom_model import CustomFluxTransformer2DModel
# from .custom_pipeline import CustomFluxPipeline, CustomFluxPipelineCfg

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

# python exps/v04sv03/test_flux_lora.py --cases 1000 --gpu_id 0

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_config(path=None):
    
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path

    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1

    return config

def initialize_pipeline(config, args):
    import sys
    sys.path.append("/home/wyb/yanbin/GradioDemo/multi_layer_sd3/exps")
    sys.path.append("/home/wyb/yanbin/GradioDemo/multi_layer_sd3/exps/v04sv03")
    from custom_model import CustomFluxTransformer2DModel
    from custom_pipeline import CustomFluxPipeline, CustomFluxPipelineCfg

    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.transformer_varient if hasattr(config, "transformer_varient") else config.pretrained_model_name_or_path, 
        subfolder="" if hasattr(config, "transformer_varient") else "transformer", 
        revision=config.revision, 
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir='/home/wyb/cache_dir'
    ) 
    # + config.get("cache_dir", None),
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config.max_layer_num
    mmdit_config = FrozenDict(mmdit_config)
    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)

    # lora pretrained lora weights
    if hasattr(config, "pretrained_lora_dir"):
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config.pretrained_lora_dir)
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora() # don't forget to unload the lora params

    # load layer_pe
    layer_pe_path = os.path.join(args.ckpt_dir, "layer_pe.pth")
    layer_pe = torch.load(layer_pe_path)
    missing_keys, unexpected_keys = transformer.load_state_dict(layer_pe, strict=False)

    pipeline_type = CustomFluxPipelineCfg if args.use_true_cfg else CustomFluxPipeline
    pipeline = pipeline_type.from_pretrained(
        config.pretrained_model_name_or_path,
        transformer=transformer,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir=config.get("cache_dir", None),
    ).to(torch.device("cuda", index=args.gpu_id))
    pipeline.enable_model_cpu_offload(gpu_id=args.gpu_id) # save vram

    pipeline.load_lora_weights(args.ckpt_dir, adapter_name="layer")

    _SET_ADAPTER_SCALE_FN_MAPPING["CustomFluxTransformer2DModel"] = _SET_ADAPTER_SCALE_FN_MAPPING["FluxTransformer2DModel"]
    for lora_name in args.fuse_lora_list:
        lora_info = lora_mapping[lora_name]
        if "weight_name" in lora_info:
            pipeline.load_lora_weights(lora_info["url"], weight_name=lora_info["weight_name"], adapter_name=lora_name)
        else:
            pipeline.load_lora_weights(lora_info["url"], adapter_name=lora_name)
    
    adapter_names = ["layer"] + args.fuse_lora_list
    pipeline.set_adapters(adapter_names, adapter_weights=[1.0]*len(adapter_names))
    # pipeline.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    # pipeline.unload_lora_weights()

    return pipeline

def get_fg_layer_box(list_layer_pt):
    list_layer_box = []
    for layer_pt in list_layer_pt:
        alpha_channel = layer_pt[:, 3:4]

        if layer_pt.shape[1] == 3:
            list_layer_box.append(
                (0, 0, layer_pt.shape[3], layer_pt.shape[2])
            )
            continue

        # Step 1: Find the non-zero indices
        _, _, rows, cols = torch.nonzero(alpha_channel + 1, as_tuple=True)

        if (rows.numel() == 0) or (cols.numel() == 0):
            # If there are no non-zero indices, we can skip this layer
            list_layer_box.append(None)
            continue

        # Step 2: Get the minimum and maximum indices for rows and columns
        min_row, max_row = rows.min().item(), rows.max().item()
        min_col, max_col = cols.min().item(), cols.max().item()

        # Step 3: Quantize the minimum values down to the nearest multiple of 16
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16

        # Step 4: Quantize the maximum values up to the nearest multiple of 16 outside of the max
        quantized_max_row = ((max_row // 16) + 1) * 16
        quantized_max_col = ((max_col // 16) + 1) * 16
        list_layer_box.append(
            (quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row)
        )
    return list_layer_box

def get_list_layer_box(list_png_images):
    list_layer_box = []
    for img in list_png_images:
        img_np = np.array(img)
        alpha_channel = img_np[:, :, -1]

        # Step 1: Find the non-zero indices
        rows, cols = np.nonzero(alpha_channel)

        if (len(rows) == 0) or (len(cols) == 0):
            # If there are no non-zero indices, we can skip this layer
            list_layer_box.append((0, 0, 0, 0))
            continue

        # Step 2: Get the minimum and maximum indices for rows and columns
        min_row, max_row = rows.min().item(), rows.max().item()
        min_col, max_col = cols.min().item(), cols.max().item()

        # Step 3: Quantize the minimum values down to the nearest multiple of 8
        quantized_min_row = (min_row // 8) * 8
        quantized_min_col = (min_col // 8) * 8

        # Step 4: Quantize the maximum values up to the nearest multiple of 8 outside of the max
        quantized_max_row = ((max_row // 8) + 1) * 8
        quantized_max_col = ((max_col // 8) + 1) * 8
        list_layer_box.append(
            (quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row)
        )
    return list_layer_box

def pngs_to_svg(list_png_images):
    list_layer_box = get_list_layer_box(list_png_images)
    assert(len(list_png_images) == len(list_layer_box))
    width, height = list_png_images[0].width, list_png_images[0].height
    img_svg = ET.Element(
       'svg', 
        {
            "width": str(width),
            "height": str(height),
             "xmlns": "http://www.w3.org/2000/svg", 
             "xmlns:svg": "http://www.w3.org/2000/svg", 
             "xmlns:xlink":"http://www.w3.org/1999/xlink"                 
        }
    )
    for img, box in zip(list_png_images, list_layer_box):
        x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        if (w == 0 or h == 0):
            continue
        img = img.crop((x, y, x+w, y+h))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue())
        ET.SubElement(
            img_svg,
            "image",
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "xlink:href": "data:image/png;base64,"+img_str.decode('utf-8')
            }
        )
    return ET.tostring(img_svg, encoding='utf-8').decode('utf-8')

lora_mapping = {
    "ideogram" : {"url": "/openseg_blob/zhaoym/sd3/work_dirs/ideogram2m_sft_flux"},
    "add_details": {"url": "Shakker-Labs/FLUX.1-dev-LoRA-add-details"},
    "mjv6": {"url": "XLabs-AI/flux-lora-collection", "weight_name": "mjv6_lora.safetensors"},
    "realign": {"url": "/openseg_blob/zhaoym/multi_layer_sd3/work_dirs/re_align_r16_m94000/checkpoint-14000"},
}

def test_one_sample(validation_box, validation_prompt, pipeline, generator, config, args, transp_vae):
    # merged_pt = batch["merged_pt"]
    # backgd_pt = batch["backgd_pt"]
    # list_layer_pt = [layer_pt for layer_pt in batch["list_layer_pt"]]
    # validation_box = get_fg_layer_box([merged_pt, backgd_pt] + list_layer_pt)

    # validation_prompt = batch["caption"]
    # # this_index = batch["poster_index"][0]
    # this_index = f"case_{idx}"

    # if None in validation_box:
    #     continue

    # print(f"Case {idx}:")
    output, rgba_output, _ = pipeline(
        prompt=validation_prompt,
        validation_box=validation_box,
        generator=generator,
        height=config.resolution,
        width=config.resolution,
        num_layers=len(validation_box),
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        sdxl_vae=transp_vae,
    )
    images = output.images   # list of PIL, len=layers
    rgba_images = [Image.fromarray(arr, 'RGBA') for arr in rgba_output]

    output_gradio = []

    # os.makedirs(os.path.join(args.save_dir, "merged"), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, "merged_rgba"), exist_ok=True)
    # os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)
    # os.system(f"rm -rf {os.path.join(args.save_dir, this_index)}/*")
    # for frame_idx, frame_pil in enumerate(images):
    #     frame_pil.save(os.path.join(args.save_dir, this_index, f"layer_{frame_idx}.png"))
    #     if frame_idx == 0:
    #         frame_pil.save(os.path.join(args.save_dir, "merged", f"{this_index}.png"))
    merged_pil = images[1].convert('RGBA')
    for frame_idx, frame_pil in enumerate(rgba_images):
        if frame_idx < 2:
            frame_pil = images[frame_idx].convert('RGBA') # merged and background
        else:
            merged_pil = Image.alpha_composite(merged_pil, frame_pil)
        # frame_pil.save(os.path.join(args.save_dir, this_index, f"layer_{frame_idx}_rgba.png"))
        output_gradio.append(frame_pil)
    
    return output_gradio

def adjust_coordinates(box):
    adjusted_box = []
    for x in box:
        # Clamp the value between 0 and 512
        clamped = max(0, min(512, x))
        # Round to the nearest multiple of 16
        if clamped % 16 != 0:
            clamped = round(clamped / 16) * 16
        adjusted_box.append(clamped)
    return tuple(adjusted_box)

def adjust_validation_box(validation_box):
    return [adjust_coordinates(box) for box in validation_box]

def gradio_test_one_sample(validation_prompt, validation_box_str, pipeline, generator, config, args, transp_vae):
    # Safely parse the string input into a list of tuples
    try:
        validation_box = ast.literal_eval(validation_box_str)
    except Exception as e:
        return [f"Error parsing validation_box: {e}"]

    # Validate that validation_box is a list of tuples, each of length 4
    if not isinstance(validation_box, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in validation_box):
        return ["validation_box must be a list of tuples, each of length 4."]

    validation_box = adjust_validation_box(validation_box)
    
    # Call your backend function
    result_images = test_one_sample(validation_box, validation_prompt, pipeline, generator, config, args, transp_vae)
    
    svg_img = pngs_to_svg(result_images[1:])

    svg_file_path = '/home/wyb/openseg_blob/v-yanbin/GradioDemo/multi_layer_sd3/image.svg'
    os.makedirs(os.path.dirname(svg_file_path), exist_ok=True)
    with open(svg_file_path, 'w', encoding='utf-8') as f:
        f.write(svg_img)
    
    return result_images, svg_file_path

def construction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v04sv03")
    parser.add_argument("--name", type=str, default="v04sv03_lora_r64_upto50layers_bs1_lr1_prodigy_800k_wds_512_filtered_10ep_none_8gpu") # v04sv03_lora_r64_upto50layers_bs1_lr1_prodigy_800k_wds_512_filtered_10ep_none_8gpu
    parser.add_argument("--ckpt_idx", type=int, default=94000) # 94000
    parser.add_argument("--fuse_lora_list", type=list, default=["ideogram"])
    parser.add_argument("--variant", type=str, default="fuse_ideo1")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--use_true_cfg", type=bool, default=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()

    args.name = "v04sv03_lora_r64_upto50layers_bs1_lr1_prodigy_800k_wds_512_filtered_10ep_none_8gpu"
    args.ckpt_idx = 94000
    args.fuse_lora_list = ["ideogram"]

    if args.seed is not None:
        seed_everything(args.seed)

    os.chdir('/home/wyb/yanbin/GradioDemo/multi_layer_sd3/')  # 替换为您的目标路径

    cfg_path = f"exps/{args.version}/configs/{args.name}.py"
    config = parse_config(cfg_path)

    if args.ckpt_idx is not None:
        args.ckpt_dir = f"/home/wyb/openseg_blob/zhaoym/multi_layer_sd3/work_dirs/{args.name}/checkpoint-{args.ckpt_idx}"
    else:
        args.ckpt_dir = f"/home/wyb/openseg_blob/zhaoym/multi_layer_sd3/work_dirs/{args.name}"

    args.save_dir = f"outputs/{args.name}"
    if args.variant is not None: args.save_dir += '_' + args.variant

    pipeline = initialize_pipeline(config, args)

    import sys
    sys.path.append("/home/wyb/yanbin/GradioDemo/multi_layer_sd3/exps")
    sys.path.append("/home/wyb/yanbin/GradioDemo/multi_layer_sd3/exps/v04sv03")
    from basecode_flux.transformer_vae import AutoencoderKLTransformerTraining as CustomVAE
    from PIL import Image
    transp_vae = CustomVAE() # by zhicong
    vae_path = '/home/wyb/openseg_blob/zhitang/transparent-vae/output/dec_8x_a100_80g_512res_flux_1/checkpoints/latest.pt'
    missing, unexpected = transp_vae.load_state_dict(torch.load(vae_path)['model'], strict=False)
    transp_vae.eval()

    generator = torch.Generator(device=torch.device("cuda", index=args.gpu_id)).manual_seed(args.seed) if args.seed else None
    
    return pipeline, generator, config ,args, transp_vae

def main():
    pipeline, generator, config ,args, transp_vae = construction()

    gradio_test_one_sample_partial = partial(
        gradio_test_one_sample,
        pipeline=pipeline,
        generator=generator,
        config=config,
        args=args,
        transp_vae=transp_vae,
    )

    def clear_inputs1():
        return ""
    
    def clear_inputs2():
        return "", ""

    with gr.Blocks() as demo:
        gr.HTML("<h2>Multi Layer SD3</h2>")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(lines=10, placeholder="Enter prompt text", label="Prompt")
                tuple_input = gr.Textbox(lines=5, placeholder="Enter list of tuples, e.g., [(1, 2, 3, 4), (5, 6, 7, 8)]", label="Validation Box")
                with gr.Row():
                    clear_btn2 = gr.Button("Clear")
                    model_btn2 = gr.Button("Commit")
                    # transfer_btn2 = gr.Button("Import from above")
                
            with gr.Column():
                result_images = gr.Gallery(label="Result Images", columns=5, height='auto')
        examples = gr.Examples(
            examples=[
                ["a dog sit on the grass", "[(0,0,512,512),(0,0,512,512),(128,64,360,448)]"]
            ],
            inputs=[text_input, tuple_input]
        )
        gr.HTML("<h1>SVG Image</h1>")
        svg_file = gr.File(label="Download SVG Image")
        svg_editor = gr.HTML(label="SVG Editor")
        
        model_btn1.click(
            fn=process_preddate, 
            inputs=[generate_method, intention_input], 
            outputs=[wholecaption_output, list_box_output, json_file], 
            api_name="process_preddate"
        )
        clear_btn1.click(
            fn=clear_inputs1, 
            inputs=[], 
            outputs=[intention_input]
        )
        model_btn2.click(
            fn=process_svg, 
            inputs=[text_input, tuple_input], 
            outputs=[result_images, svg_file, svg_editor], 
            api_name="process_svg"
        )
        clear_btn2.click(
            fn=clear_inputs2, 
            inputs=[], 
            outputs=[text_input, tuple_input]
        )

    # Launch
    demo.launch(allowed_paths=["/home/wyb/yanbin"], share=True)

if __name__ == "__main__":
    main()