from qvic.model.builder import load_pretrained_model
from qvic.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from qvic.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from qvic.conversation import conv_templates
import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
    
warnings.filterwarnings("ignore")

#----- Model settings -----

pretrained = "ckpt/QViC-MF-7B"  # TODO: switch back to "Fujitsu/QViC-MF-7B" once published on HF
model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
conv_template="qwen_1_5"

#----- Task settings -----

video_path = "playground/demo/xU25MMA2N4aVtYay.mp4"
# Please describe this video in detail.

max_frames_num = 64 # -1: all frames (fps sampling)
context_memory_length = 256
max_new_tokens = 1024 #4096
fps = 1 #2

torch_dtype = "bfloat16"
attn_implementation = "sdpa"

#video, image_sizes, modalities, video_time, frame_time, frame_idx = load_video_tensor(video_path, max_frames_num)

#-----

model_name = get_model_name_from_path(pretrained)
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, model_base, model_name, torch_dtype=torch_dtype, device_map=device_map, attn_implementation=attn_implementation)

model.eval()
model.set_context_memory_length(context_memory_length)

#----- Debug settings -----
#model.set_debug_drawing_memory(True)
#----- Debug settings -----

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if (len(frame_idx) > max_frames_num or force_sample) and max_frames_num > 0:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time, frame_idx


def load_video_tensor(video_path, max_frames_num=max_frames_num, fps=fps):
    video_, frame_time, video_time, frame_idx = load_video(video_path, max_frames_num, fps=fps, force_sample=True)
    
    video = image_processor.preprocess(video_, return_tensors="pt")["pixel_values"].cuda()
    if torch_dtype == "bfloat16":
        video = video.bfloat16()
    else:
        video = video.half()
    video = [video]
    image_sizes = None
    modalities=["video"]
    
    print(f"Video loaded: {video_path}")
    print(f"Video time: {video_time:.2f} seconds")
    print(f"Frame time: {frame_time}")
    print(f"Number of frames: {len(video[0])}")
    print(f"video[0].shape: {video[0].shape}")
    print(f"image_sizes: {image_sizes}")
        
    return video, image_sizes, modalities, video_time, frame_time, frame_idx


video, image_sizes, modalities, video_time, frame_time, frame_idx = load_video_tensor(video_path, max_frames_num)


def make_input_ids(query, add_instruction=False, add_assitant_message=True):
    if add_instruction:
        instruciton = f"Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{instruciton}\n{query}"
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    if add_assitant_message:
        conv.append_message(conv.roles[1], None)

    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    return input_ids, question

    
while True:
    
    print("\n **Input query** \n")
    query = input()
    
    if query == "b":
        breakpoint()
        
    query = query.replace("\\n", "\n")

    input_ids, question = make_input_ids(query)
    input_ids_q, _ = make_input_ids(query)

    print("Video path:\n", video_path)
    print("Question:\n", question)

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=video,
            image_sizes=image_sizes,
            modalities=modalities,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            input_ids_q=input_ids_q,
            disable_tqdm=False,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    print("Video path:\n", video_path)
    print("Question:\n", question)
    print("Answer:\n", text_outputs)
