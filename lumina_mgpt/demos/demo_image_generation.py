import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

import argparse
import builtins
import datetime
import multiprocessing as mp
import traceback
from typing import List, Optional

import gradio as gr
import torch

from data.item_processor import generate_crop_size_list
from inference_solver import FlexARInferenceSolver
from xllmx.util.misc import random_seed


class Ready:
    pass


class ModelFailure:
    pass


@torch.no_grad()
def model_worker(
    rank: int,
    args: argparse.Namespace,
    barrier: mp.Barrier,
    request_queue: mp.Queue,
    response_queue: Optional[mp.Queue] = None,
) -> None:
    """
    The worker function that manipulates the GPU to run the inference.
    Exact n_gpu workers are started, with each one operating on a separate GPU.

    Args:
        rank (int): Distributed rank of the worker.
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    builtin_print = builtins.print

    def print(*args, **kwargs):
        kwargs["flush"] = True
        now = datetime.datetime.now().time()
        builtin_print("[{}] ".format(now), end="")  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print

    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    # dist.init_process_group(
    #     backend="nccl", rank=rank, world_size=world_size,
    #     init_method=f"tcp://{args.master_addr}:{args.master_port}",
    # )
    # print(f"| distributed init on worker {rank}/{world_size}. "
    #       f"using gpu: {gpu_id}")
    torch.cuda.set_device(gpu_id)

    inference_solver = FlexARInferenceSolver(
        model_path=args.pretrained_path,
        precision=args.precision,
    )

    barrier.wait()

    while True:
        if response_queue is not None:
            response_queue.put(Ready())
        try:
            prompt, resolution, seed, gen_t, cfg, image_top_k = request_queue.get()

            random_seed(seed=seed)

            prompt = f"Generate an image of {resolution} according to the following prompt:\n{prompt}"
            print(prompt)

            generated = inference_solver.generate(
                [],
                [[prompt, None]],
                5000,
                gen_t,
                logits_processor=inference_solver.create_logits_processor(
                    cfg=cfg, text_top_k=5, image_top_k=image_top_k
                ),
            )

            print("*" * 100)
            print(generated[1])

            stream_response = {"text": generated[0], "image": generated[1], "prompt": prompt, "end_of_content": True}

            print(generated[1])

            if response_queue is not None:
                print("here")
                response_queue.put(stream_response)

        except Exception:
            print(traceback.format_exc())
            response_queue.put(ModelFailure())


def gradio_worker(
    request_queues: List[mp.Queue],
    response_queue: mp.Queue,
    args: argparse.Namespace,
    barrier: mp.Barrier,
) -> None:
    """
    The gradio worker is responsible for displaying the WebUI and relay the
    requests to model workers. It should be launched only once.

    Args:
        request_queues (List[mp.Queue]): A list of request queues (one for
            each model worker).
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    def check_input_sanity(text_input: str):
        if len(text_input) > 1024:
            raise gr.Error("please do not send more than 1024 characters to this demo")
        if text_input.count("<|image|>") != 0:
            raise gr.Error("please do not send <|image|> tokens to this demo")

    def stream_model_output(prompt, resolution, seed, gen_t, cfg, image_top_k):

        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, Ready):
                break
        for queue in request_queues:
            queue.put((prompt, resolution, seed, gen_t, cfg, image_top_k))
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, ModelFailure):
                raise RuntimeError
            if content_piece["end_of_content"]:
                yield content_piece["image"][0], content_piece["prompt"]
                break
            else:
                yield None, None

    def show_real_prompt():
        return gr.update(visible=True)

    with gr.Blocks(css="#image_input {height: 100% !important}") as demo:
        gr.Markdown("# Lumina-mGPT Image Generation Demo\n")
        with gr.Row() as r:
            with gr.Column(scale=1):
                prompt = gr.Textbox(lines=3, interactive=True, label="Prompt")
                with gr.Row():
                    patch_size = 32
                    res_choices = generate_crop_size_list((args.target_size // patch_size) ** 2, patch_size)
                    res_choices = [f"{w}x{h}" for w, h in res_choices]
                    assert f"{args.target_size}x{args.target_size}" in res_choices
                    resolution = gr.Dropdown(
                        value=f"{args.target_size}x{args.target_size}", choices=res_choices, label="Resolution"
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        seed = gr.Slider(
                            minimum=0,
                            maximum=int(1e5),
                            value=300,
                            step=1,
                            interactive=True,
                            label="Seed (0 for random)",
                        )
                    with gr.Column(scale=1):
                        gen_t = gr.Slider(
                            minimum=0.0,
                            maximum=4.0,
                            value=1.0,
                            interactive=True,
                            label="gen_t",
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        cfg = gr.Slider(
                            minimum=0.0,
                            maximum=16.0,
                            value=3.0,
                            interactive=True,
                            label="cfg",
                        )
                    with gr.Column(scale=1):
                        image_top_k = gr.Slider(
                            minimum=0,
                            maximum=8192,
                            value=4000,
                            interactive=True,
                            label="Image Top-k",
                        )
                submit_button = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_img = gr.Image(
                    label="Generated image",
                    interactive=False,
                )
                real_prompt = gr.Textbox(
                    label="Real Prompt", interactive=False, visible=False, show_label=True, show_copy_button=True
                )

        prompt.submit(check_input_sanity, [prompt], []).success(
            stream_model_output, [prompt, resolution, seed, gen_t, cfg, image_top_k], [output_img, real_prompt]
        )
        submit_button.click(check_input_sanity, [prompt], []).success(
            stream_model_output, [prompt, resolution, seed, gen_t, cfg, image_top_k], [output_img, real_prompt]
        ).success(show_real_prompt, [], [real_prompt])
    barrier.wait()
    demo.queue(api_open=True).launch(
        share=True,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("X-LLM-X Chat Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        help="A list of space-separated gpu ids to run the model on. "
        "The model will span across GPUs in tensor-parallel mode.",
    )
    group.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="Number of GPUs to run the model on. Equivalent to " "--gpu_ids 0 1 2 ... n-1",
    )
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the model checkpoints.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help="The dtype used for model weights and inference.",
    )
    parser.add_argument(
        "--target_size", type=int, default=768, choices=[512, 768, 1024], help="The target image generation size."
    )
    args = parser.parse_args()

    # check and setup gpu_ids to use
    if args.gpu_ids is None:
        if args.n_gpus is None:
            args.n_gpus = 1
        assert args.n_gpus > 0, "The demo currently must run on a positive number of GPUs."
        args.gpu_ids = list(range(args.n_gpus))

    assert len(args.gpu_ids) == 1, "Currently only supports running on a single GPU."

    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    mp.set_start_method("spawn")

    # setup the queues and start the model workers
    request_queues = []
    response_queue = mp.Queue()
    worker_processes = []
    barrier = mp.Barrier(len(args.gpu_ids) + 1)
    for rank, gpu_id in enumerate(args.gpu_ids):
        request_queue = mp.Queue()
        rank_response_queue = response_queue if rank == 0 else None
        process = mp.Process(
            target=model_worker,
            args=(rank, args, barrier, request_queue, rank_response_queue),
        )
        process.start()
        worker_processes.append(process)
        request_queues.append(request_queue)

    gradio_worker(request_queues, response_queue, args, barrier)
