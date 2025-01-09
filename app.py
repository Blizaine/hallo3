import os
import shutil
from huggingface_hub import snapshot_download
import gradio as gr
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scripts.inference_hallo3 import inference_process_hallo3  # Adjusted for Hallo3
import argparse
import uuid

is_shared_ui = True if "fudan-generative-ai/hallo3" in os.environ['SPACE_ID'] else False

if not is_shared_ui:
    hallo3_dir = snapshot_download(repo_id="fudan-generative-ai/hallo3", local_dir="pretrained_models_hallo3")

def run_inference(source_image, driving_audio, progress=gr.Progress(track_tqdm=True)):
    if is_shared_ui:
        raise gr.Error("This Space only works in duplicated instances")
        
    unique_id = uuid.uuid4()
    
    args = argparse.Namespace(
        config='configs/inference_hallo3/default.yaml',  # Updated config for Hallo3
        source_image=source_image,
        driving_audio=driving_audio,
        output=f'output-{unique_id}.mp4',
        pose_weight=1.2,  # Adjusted default weights for Hallo3
        face_weight=1.2,
        lip_weight=1.0,
        face_expand_ratio=1.3,  # Adjusted for Hallo3 requirements
        checkpoint="pretrained_models_hallo3/model.pth"  # Specific checkpoint for Hallo3
    )
    
    inference_process_hallo3(args)  # Updated inference process for Hallo3
    return f'output-{unique_id}.mp4'

css = '''
/* Custom CSS remains the same */
'''

with gr.Blocks(css=css) as demo:
    if is_shared_ui:
        top_description = gr.HTML(f'''
            <div class="gr-prose">
                <h2 class="custom-color">Attention: this Space needs to be duplicated to work</h2>
                <p class="main-message custom-color">
                    To make it work, <strong>duplicate the Space</strong> and run it on your own profile using a <strong>private</strong> GPU.
                </p>
            </div>
        ''', elem_id="warning-duplicate")
    gr.Markdown("# Demo for Hallo3: Advanced Audio-Driven Portrait Animation")
    gr.Markdown("Generate advanced talking head avatars driven by audio with improved realism and dynamics using Hallo3.")
    gr.Markdown("""
Hallo3 has updated requirements for input data:

For the source image:

1. High resolution is recommended for improved output.
2. The face should occupy 50%-70% of the image.
3. Minimal rotation for best results.

For the driving audio:

1. Must be in WAV format.
2. Ensure vocals are clear, even with background music.

For samples and more information, visit the [official page](https://huggingface.co/spaces/fudan-generative-ai/hallo3).
                """)
    with gr.Row():
        with gr.Column():
            avatar_face = gr.Image(type="filepath", label="Face")
            driving_audio = gr.Audio(type="filepath", label="Driving Audio")
            generate = gr.Button("Generate")
        with gr.Column():
            output_video = gr.Video(label="Generated Talking Head")

    generate.click(
        fn=run_inference,
        inputs=[avatar_face, driving_audio],
        outputs=output_video
    )
    
demo.launch()
