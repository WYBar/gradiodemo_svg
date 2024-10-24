from multi_layer_sd3.exps.v04sv03.test_gradio import gradio_test_one_sample, construction
from LLM_For_Layout_Planning.inference import inference, construction_layout
import gradio as gr
from functools import partial
import requests
import base64
import os
import xml.etree.cElementTree as ET
import PIL.Image as Image
import time
import random

def generate_unique_filename():
    # 生成一个基于时间戳和随机数的唯一文件名
    timestamp = int(time.time() * 1000)  # 时间戳，毫秒级
    random_num = random.randint(1000, 9999)  # 随机数
    unique_filename = f"{timestamp}_{random_num}"
    return unique_filename

def upload_to_github(file_path, 
                     repo='WYBar/gradiodemo_svg', 
                     branch='main', 
                     token='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'):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'rb') as file:
        content = file.read()
    encoded_content = base64.b64encode(content).decode('utf-8')
    unique_filename = generate_unique_filename()
    url = f"https://api.github.com/repos/{repo}/contents/{unique_filename}.svg"
    headers = {
        "Authorization": f"token {token}"
    }
    response = requests.get(url, headers=headers)
    
    sha = None
    if response.status_code == 200:
        sha = response.json()['sha']
    elif response.status_code == 404:
        # 文件不存在，不需要SHA
        pass
    else:
        print(f"Failed to get file status: {response.status_code}")
        print(response.text)
        return None
    
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    data = {
        "message": "upload svg file",
        "content": encoded_content,
        "branch": branch
    }
    
    if sha:
        # 文件存在，更新文件
        print('sha exists')
        data["sha"] = sha
        response = requests.put(url, headers=headers, json=data)
    else:
        # 文件不存在，创建新文件
        print("sha not exist")
        response = requests.put(url, headers=headers, json=data)
        
    print(response.status_code)
    print(response.text)
    if response.status_code in [200, 201]:
        print(response.json()['content']['download_url'])
        return response.json()['content']['download_url']
    else:
        return None

def main():
    model, quantizer, tokenizer, width, height = construction_layout()
    
    inference_partial = partial(
        inference,
        model=model,
        quantizer=quantizer,
        tokenizer=tokenizer,
        width=width,
        height=height
    )
    
    def process_preddate(generate_method, intention):
        preddata, json_file = inference_partial(generate_method, intention)
        wholecaption = preddata["wholecaption"]
        layouts = preddata["layout"]
        list_box = []
        for i, layout in enumerate(layouts):
            x, y = layout["x"], layout["y"]
            width, height = layout["width"], layout["height"]
            if i == 0:
                list_box.append((0, 0, width, height))
                list_box.append((0, 0, width, height))
            else:
                left = x - width // 2
                top = y - height // 2
                right = x + width // 2
                bottom = y + height // 2
                list_box.append((left, top, right, bottom))
            
        return wholecaption, str(list_box), json_file
    
    # def process_preddate(generate_method, intention):
    #     list_box = [(0, 0, 512, 512), (0, 0, 512, 512), (136, 184, 512, 512), (144, 0, 512, 512), (0, 0, 328, 136), (160, 112, 512, 360), (168, 112, 512, 360), (40, 232, 112, 296), (32, 88, 248, 176), (48, 424, 144, 448), (48, 464, 144, 488), (240, 464, 352, 488), (384, 464, 488, 488), (48, 480, 144, 504), (240, 480, 360, 504), (456, 0, 512, 56), (0, 0, 56, 40), (440, 0, 512, 40), (0, 24, 48, 88), (48, 168, 168, 240)]
    #     wholecaption = "Design an engaging and vibrant recruitment advertisement for our company. The image should feature three animated characters in a modern cityscape, depicting a dynamic and collaborative work environment. Incorporate a light bulb graphic with a question mark, symbolizing innovation, creativity, and problem-solving. Use bold text to announce \"WE ARE RECRUITING\" and provide the company's social media handle \"@reallygreatsite\" and a contact phone number \"+123-456-7890\" for interested individuals. The overall design should be playful and youthful, attracting potential recruits who are innovative and eager to contribute to a lively team."
    #     json_file = "/home/wyb/openseg_blob/v-yanbin/GradioDemo/LLM-For-Layout-Planning/inference_test.json"
    #     return wholecaption, str(list_box), json_file

    pipeline, generator, config ,args, transp_vae = construction()

    gradio_test_one_sample_partial = partial(
        gradio_test_one_sample,
        pipeline=pipeline,
        generator=generator,
        config=config,
        args=args,
        transp_vae=transp_vae,
    )
    
    def process_svg(text_input, tuple_input):
        result_images = []
        # print(text_input, tuple_input)
        result_images, svg_file_path = gradio_test_one_sample_partial(text_input, tuple_input)
        # print(text_input, tuple_input)
        url = upload_to_github(file_path=svg_file_path)
        if url != None:
            print(f"File uploaded to: {url}")
            svg_editor = f"""
                <iframe src="https://svgedit.netlify.app/editor/index.html?\
                storagePrompt=false&url={url}" \
                width="100%", height="800px"></iframe>
            """
        else:
            print('upload_to_github FAILED!')
            svg_editor = f"""
                <iframe src="https://svgedit.netlify.app/editor/index.html" \
                width="100%", height="800px"></iframe>
            """
        # print(f'result_images: {result_images}')
        time.sleep(3)
        
        return result_images, svg_file_path, svg_editor
    
    def one_click_generate(generate_method, intention_input):
        # 首先调用process_preddate
        wholecaption_output, list_box_output, json_file = process_preddate(generate_method, intention_input)
        
        # 然后将process_preddate的输出作为process_svg的输入
        result_images, svg_file, svg_editor = process_svg(wholecaption_output, list_box_output)
        
        # 返回两个函数的输出
        return wholecaption_output, list_box_output, result_images, svg_file, svg_editor

    def clear_inputs1():
        return ""
    
    def clear_inputs2():
        return "", ""
    
    def transfer_inputs(wholecaption, list_box):
        return wholecaption, list_box
        
    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center;'>Gradio Demo</h1>")
        gr.HTML("<h2>LLM For Layout Planning</h2>")
        with gr.Row():
            with gr.Column():
                generate_method = gr.Radio(["v1", "v3"], label="Choose the generate method", value="v1")
                intention_input = gr.Textbox(lines=10, placeholder="Enter intention", label="Prompt")
                with gr.Row():
                    clear_btn1 = gr.Button("Clear")
                    model_btn1 = gr.Button("Commit")
                    transfer_btn1 = gr.Button("Export to below")
                    
                one_click_btn = gr.Button("One Click Generate ALL")
        
            with gr.Column():
                wholecaption_output = gr.Textbox(lines=5, placeholder="Whole Caption", label="Whole Caption")
                list_box_output = gr.Textbox(lines=5, placeholder="Validation Box", label="Validation Box")
                json_file = gr.File(label="Download JSON File")
                
        examples = gr.Examples(
            examples=[
                ["Design an engaging and vibrant recruitment advertisement for our company. The image should feature three animated characters in a modern cityscape, depicting a dynamic and collaborative work environment. Incorporate a light bulb graphic with a question mark, symbolizing innovation, creativity, and problem-solving. Use bold text to announce \"WE ARE RECRUITING\" and provide the company's social media handle \"@reallygreatsite\" and a contact phone number \"+123-456-7890\" for interested individuals. The overall design should be playful and youthful, attracting potential recruits who are innovative and eager to contribute to a lively team."]
            ],
            inputs=[intention_input]
        )
        
        gr.HTML("<h2>Multi Layer SD3</h2>")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(lines=10, placeholder="Enter prompt text", label="Prompt")
                tuple_input = gr.Textbox(lines=5, placeholder="Enter list of tuples, e.g., [(1, 2, 3, 4), (5, 6, 7, 8)]", label="Validation Box")
                with gr.Row():
                    clear_btn2 = gr.Button("Clear")
                    model_btn2 = gr.Button("Commit")
                    transfer_btn2 = gr.Button("Import from above")
                
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
        transfer_btn1.click(
            fn=transfer_inputs, 
            inputs=[wholecaption_output, list_box_output], 
            outputs=[text_input, tuple_input]
        )
        transfer_btn2.click(
            fn=transfer_inputs, 
            inputs=[wholecaption_output, list_box_output], 
            outputs=[text_input, tuple_input]
        )
        one_click_btn.click(
            fn=one_click_generate, 
            inputs=[generate_method, intention_input], 
            outputs=[wholecaption_output, list_box_output, result_images, svg_file, svg_editor]
        )
    demo.launch(allowed_paths=["/home/wyb/openseg_blob/v-yanbin/GradioDemo/LLM-For-Layout-Planning", \
        "/home/wyb/openseg_blob/v-yanbin/GradioDemo/multi_layer_sd3"], share=True)

if __name__ == "__main__":
    main()
    print("yes")
    