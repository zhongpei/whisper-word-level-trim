import torch
import gradio as gr
import json
import ffmpeg
import os
from pathlib import Path
import time
from transformers import pipeline

# checkpoint = "openai/whisper-large-v2"
# checkpoint = "openai/whisper-medium"
checkpoint = "openai/whisper-small"

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    from transformers import (
        AutomaticSpeechRecognitionPipeline,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        checkpoint).to("cuda").half()

    processor = WhisperProcessor.from_pretrained(checkpoint)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=4,
        torch_dtype=torch.float16,
        device="cuda:0"
    )
else:
    pipe = pipeline(model=checkpoint)

# TODO: no longer need to set these manually once the models have been updated on the Hub
alignment_heads = {
    "openai/whisper-tiny": [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
    "openai/whisper-tiny.en": [[1, 0], [2, 0], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    "openai/whisper-base": [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]],

    "openai/whisper-base.en": [[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]],

    "openai/whisper-small": [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]],

    "openai/whisper-small.en": [[6, 6], [7, 0], [7, 3], [7, 8], [8, 2], [8, 5], [8, 7], [9, 0], [9, 4], [9, 8], [9, 10],
                         [10, 0], [10, 1], [10, 2], [10, 3], [10, 6], [10, 11], [11, 2], [11, 4]],

    "openai/whisper-medium": [[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]],

    "openai/whisper-medium.en": [[11, 4], [14, 1], [14, 12], [14, 14], [15, 4], [16, 0], [16, 4], [16, 9], [17, 12], [17, 14],
                          [18, 7], [18, 10], [18, 15], [20, 0], [20, 3], [20, 9], [20, 14], [21, 12]],

    "openai/whisper-large-v1": [[9, 19], [11, 2], [11, 4], [11, 17], [22, 7], [22, 11], [22, 17], [23, 2], [23, 15]],

    "openai/whisper-large-v2": [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], [18, 4], [18, 11],
                         [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], [22, 9], [22, 12], [23, 5], [23, 7], [23, 13],
                         [25, 5], [26, 1], [26, 12], [27, 15]]
}

pipe.model.generation_config.alignment_heads = alignment_heads[checkpoint]

# whisper-tiny
# pipe.model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
# whisper-base
# pipe.model.generation_config.alignment_heads = [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]]
# whisper-small
# pipe.model.generation_config.alignment_heads = [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]]


videos_out_path = Path("./videos_out")
videos_out_path.mkdir(parents=True, exist_ok=True)

samples_data = sorted(Path('examples').glob('*.json'))
SAMPLES = []
for file in samples_data:
    with open(file) as f:
        sample = json.load(f)
    SAMPLES.append(sample)
VIDEOS = list(map(lambda x: [x['video']], SAMPLES))


async def speech_to_text(video_in):
    """
    Takes a video path to convert to audio, transcribe audio channel to text and char timestamps

    Using https://huggingface.co/tasks/automatic-speech-recognition pipeline
    """
    video_in = video_in[0] if isinstance(video_in, list) else video_in
    if (video_in == None):
        raise ValueError("Video input undefined")

    video_path = Path(video_in.name)
    try:
        # convert video to audio 16k using PIPE to audio_memory
        audio_memory, _ = ffmpeg.input(video_path).output(
            '-', format="wav", ac=1, ar=pipe.feature_extractor.sampling_rate).overwrite_output().global_args(
            '-loglevel', 'quiet').run(capture_stdout=True)
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        print(f'Transcribing via local model')
        output = pipe(audio_memory, chunk_length_s=10,
                      stride_length_s=[4, 2], return_timestamps="word")
        transcription = output["text"]
        chunks = output["chunks"]
        timestamps_var = [{"word": chunk["text"], "timestamp": (
            chunk["timestamp"][0], chunk["timestamp"][1]), "state": True} for chunk in chunks]

        words = [(word['word'], '+' if word['state'] else '-')
                 for word in timestamps_var]
        return (words, timestamps_var, video_in.name)
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


async def cut_timestamps_to_video(video_in, timestamps_var):
    video_in = video_in[0] if isinstance(video_in, list) else video_in
    if (video_in == None or timestamps_var == None):
        raise ValueError("Inputs undefined")

    video_path = Path(video_in.name)
    video_file_name = video_path.stem

    timestamps_to_cut = [
        (timestamps_var[i]['timestamp'][0], timestamps_var[i]['timestamp'][1])
        for i in range(len(timestamps_var)) if timestamps_var[i]['state']]

    between_str = '+'.join(
        map(lambda t: f'between(t,{t[0]},{t[1]})', timestamps_to_cut))

    if timestamps_to_cut:
        video_file = ffmpeg.input(video_path)
        video = video_file.video.filter(
            "select", f'({between_str})').filter("setpts", "N/FRAME_RATE/TB")
        audio = video_file.audio.filter(
            "aselect", f'({between_str})').filter("asetpts", "N/SR/TB")

        output_video = f'./videos_out/{video_file_name}.mp4'
        print(f'Cutting video to {output_video}')
        ffmpeg.concat(video, audio, v=1, a=1).output(
            output_video).overwrite_output().global_args().run()
    else:
        output_video = video_path

    return output_video


css = """
#words-container {
    max-height: 400px;
    overflow-y: scroll !important;
}
"""
with gr.Blocks(css=css) as demo:
    timestamps_var = gr.JSON(visible=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            # Whisper: Word-Level Video Trimming
            Quick edit a video by trimming out words.
            Using the [Huggingface Automatic Speech Recognition Pipeline](https://huggingface.co/tasks/automatic-speech-recognition)
            with [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
            """)

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(
                label="Upload Video File", file_count=1, scale=1)
            video_preview = gr.Video(
                label="Video Preview", scale=3, intervactive=False)
            # with gr.Row():
            #     transcribe_btn = gr.Button(
            #         "Transcribe Audio")

        with gr.Column():
            text_in = gr.HighlightedText(
                label="Transcription", combine_adjacent=False, show_legend=True, color_map={"+": "green", "-": "red"},
                elem_id="words-container")
            with gr.Row():
                cut_btn = gr.Button("Cut Video")
                select_all_words = gr.Button("Select All Words")
                reset_words = gr.Button("Reset Words")
            video_out = gr.Video(label="Video Out")




    def select_text(evt: gr.SelectData, timestamps_var):
        index = evt.index
        timestamps_var[index]['state'] = not timestamps_var[index]['state']
        words = [(word['word'], '+' if word['state'] else '-')
                 for word in timestamps_var]
        return timestamps_var, words


    def words_selection(timestamps_var, reset=False):
        if reset:
            for word in timestamps_var:
                word['state'] = True
        else:
            # reverse the state of all words
            for word in timestamps_var:
                word['state'] = False

        words = [(word['word'], '+' if word['state'] else '-')
                 for word in timestamps_var]
        return timestamps_var, words


    file_upload.upload(speech_to_text, inputs=[file_upload], outputs=[
        text_in, timestamps_var, video_preview])
    select_all_words.click(words_selection, inputs=[timestamps_var], outputs=[
        timestamps_var, text_in], queue=False, show_progress=False)
    reset_words.click(lambda x: words_selection(x, True), inputs=[timestamps_var], outputs=[
        timestamps_var, text_in], queue=False, show_progress=False)
    text_in.select(select_text, inputs=timestamps_var,
                   outputs=[timestamps_var, text_in], queue=False, show_progress=False)
    # transcribe_btn.click(speech_to_text, inputs=[file_upload], outputs=[
    #                      text_in, transcription_var, timestamps_var, video_preview])
    cut_btn.click(cut_timestamps_to_video, [
        file_upload, timestamps_var], [video_out])

demo.queue()
if __name__ == "__main__":
    demo.launch(debug=True, server_name='0.0.0.0', enable_queue=False, share=False)
