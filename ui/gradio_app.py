"""
Quick demo UI. Run directly (uses the pipeline in-process, no need for the
FastAPI server to be running):

    python -m ui.gradio_app
"""
import os
import gradio as gr

from src.pipeline import MomentRetrievalPipeline

GROUNDING_CKPT = os.environ.get("GROUNDING_CKPT")  # None -> coarse-window-only demo
pipeline = MomentRetrievalPipeline(grounding_ckpt=GROUNDING_CKPT)


def search(video_file, query):
    use_grounding = GROUNDING_CKPT is not None
    preds = pipeline.run(video_file, query, use_grounding=use_grounding)
    rows = [[f"{p.start_sec:.1f}s", f"{p.end_sec:.1f}s", f"{p.confidence:.3f}"] for p in preds]
    return rows


with gr.Blocks(title="Video Moment Retrieval") as demo:
    gr.Markdown("# Find a moment in a video with natural language")
    with gr.Row():
        video_input = gr.Video(label="Upload video")
        query_input = gr.Textbox(label="Query", placeholder="e.g. the dog jumps into the pool")
    run_btn = gr.Button("Search")
    output_table = gr.Dataframe(headers=["start", "end", "confidence"], label="Predicted moments")

    run_btn.click(fn=search, inputs=[video_input, query_input], outputs=output_table)

if __name__ == "__main__":
    demo.launch()
