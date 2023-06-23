import gradio as gr
import json
import ffmpeg
import os
from pathlib import Path
import time
from transformers import pipeline
import torch

# checkpoint = "openai/whisper-tiny"
# checkpoint = "openai/whisper-base"
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
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=8,
        torch_dtype=torch.float16,
        device="cuda:0"
    )
else:
    pipe = pipeline(model=checkpoint)


# TODO: no longer need to set these manually once the models have been updated on the Hub
# whisper-tiny
# pipe.model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
# whisper-base
# pipe.model.generation_config.alignment_heads = [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]]
# whisper-small
pipe.model.generation_config.alignment_heads = [[5, 3], [5, 9], [
    8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]]


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
            '-', format="wav", ac=1, ar=pipe.feature_extractor.sampling_rate).overwrite_output().global_args('-loglevel', 'quiet').run(capture_stdout=True)
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        print(f'Transcribing via local model')
        output = pipe(audio_memory, chunk_length_s=10,
                      stride_length_s=[4, 2], return_timestamps="word")
        transcription = output["text"]
        chunks = output["chunks"]
        timestamps_var = [{"word": chunk["text"], "timestamp":(
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
        ffmpeg.concat(video, audio, v=1, a=1).output(
            output_video).overwrite_output().global_args('-loglevel', 'quiet').run()
    else:
        output_video = video_path

    return output_video


with gr.Blocks() as demo:
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
                label="Transcription", combine_adjacent=False, show_legend=True, color_map={"+": "green", "-": "red"})
            with gr.Row():
                cut_btn = gr.Button("Cut Video")
                select_all_words = gr.Button("Select All Words")
                reset_words = gr.Button("Reset Words")
            video_out = gr.Video(label="Video Out")
    with gr.Row():
        gr.Examples(
            fn=speech_to_text,
            examples=["./examples/ShiaLaBeouf.mp4",
                      "./examples/zuckyuval.mp4",
                      "./examples/cooking.mp4"],
            inputs=[file_upload],
            outputs=[text_in, timestamps_var, video_preview],
            cache_examples=True)

    with gr.Row():
        gr.Markdown("""
        #### Video Credits

        1. [Cooking](https://vimeo.com/573792389)
        1. [Shia LaBeouf "Just Do It"](https://www.youtube.com/watch?v=n2lTxIk_Dr0)
        1. [Mark Zuckerberg & Yuval Noah Harari in Conversation](https://www.youtube.com/watch?v=Boj9eD0Wug8)
        """)

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
    demo.launch(debug=True)
