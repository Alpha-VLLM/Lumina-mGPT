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

from inference_solver import FlexARInferenceSolver
from xllmx.util.misc import random_seed


class Ready:
    pass


class ModelFailure:
    pass


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
        model_path=args.pretrained_path, precision=args.precision, target_size=args.target_size
    )

    barrier.wait()

    while True:
        if response_queue is not None:
            response_queue.put(Ready())
        try:
            existing_images, chatbot, max_gen_len, seed, gen_t, cfg, image_top_k, text_top_k = request_queue.get()

            print(chatbot)

            random_seed(seed=seed)

            generated = inference_solver.generate(
                existing_images,
                chatbot,
                max_gen_len,
                gen_t,
                logits_processor=inference_solver.create_logits_processor(
                    cfg=cfg, text_top_k=text_top_k, image_top_k=image_top_k
                ),
            )

            stream_response = {"text": generated[0], "image": generated[1], "end_of_content": True}
            print(generated[1])
            if response_queue is not None:
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

    def check_input_sanity(text_input: str, new_images):
        if new_images is None:
            new_images = []

        print(new_images)

        if text_input.count("<|image|>") != len(new_images):
            raise gr.Error("please make sure that you have the same number of image inputs and <|image|> tokens")

    def show_user_input(text_input, new_images, chatbot, chatbot_display, existing_images):

        existing_images = [] if existing_images is None else existing_images
        new_images = [] if new_images is None else new_images

        return (
            "",
            [],
            chatbot + [[text_input, None]],
            chatbot_display + [[text_input, None]],
            existing_images + new_images,
        )

    def stream_model_output(
        existing_images, chatbot, chatbot_display, max_gen_len, seed, gen_t, cfg, image_top_k, text_top_k
    ):

        existing_images = [] if existing_images is None else existing_images

        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, Ready):
                break
        for queue in request_queues:
            queue.put(
                ([_[0] for _ in existing_images], chatbot, max_gen_len, seed, gen_t, cfg, image_top_k, text_top_k)
            )
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, ModelFailure):
                raise RuntimeError
            chatbot_display[-1][1] = content_piece["text"].replace("<", "&lt;").replace(">", "&gt;")
            if content_piece["end_of_content"]:
                chatbot[-1][1] = content_piece["text"]
                chatbot_display[-1][1] = content_piece["text"]
                yield chatbot, chatbot_display, existing_images + [(_, None) for _ in content_piece["image"]]
                break
            else:
                yield chatbot, chatbot_display, []

    def clear():
        chatbot = []
        chatbot_display = []
        text_input = ""
        return chatbot, chatbot_display, text_input

    with gr.Blocks(css="#image_input {height: 100% !important}") as demo:
        gr.Markdown("# Lumina-mGPT Demo\n")
        with gr.Row() as r:
            with gr.Column(scale=1):
                existing_images = gr.Gallery(value=[], label="Existing Images", interactive=False)
                chatbot = gr.Chatbot(visible=False)
                chatbot_display = gr.Chatbot()
            with gr.Column(scale=1):
                new_images = gr.Gallery(value=[], label="Image Inputs", interactive=True)
                text_input = gr.Textbox()
                submit_button = gr.Button("Submit", variant="primary")
                clear_button = gr.ClearButton([existing_images, chatbot, chatbot_display, text_input, new_images])
                with gr.Row():
                    with gr.Column(scale=1):
                        max_gen_len = gr.Slider(
                            minimum=1,
                            maximum=5000,
                            value=2048,
                            interactive=True,
                            label="max new tokens",
                        )
                    with gr.Column(scale=1):
                        seed = gr.Slider(
                            minimum=0,
                            maximum=int(1e5),
                            value=1,
                            step=1,
                            interactive=True,
                            label="Seed (0 for random)",
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_t = gr.Slider(
                            minimum=0.0,
                            maximum=4.0,
                            value=1.0,
                            interactive=True,
                            label="gen_t",
                        )
                    with gr.Column(scale=1):
                        cfg = gr.Slider(
                            minimum=0.0,
                            maximum=16.0,
                            value=1.0,
                            interactive=True,
                            label="cfg",
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_top_k = gr.Slider(
                            minimum=0,
                            maximum=8192,
                            value=2000,
                            interactive=True,
                            label="Image Top-k",
                        )
                    with gr.Column(scale=1):
                        text_top_k = gr.Slider(
                            minimum=0,
                            maximum=9999,
                            value=5,
                            interactive=True,
                            label="Text Top-k",
                        )

        text_input.submit(check_input_sanity, [text_input, new_images], []).success(
            show_user_input,
            [text_input, new_images, chatbot, chatbot_display, existing_images],
            [text_input, new_images, chatbot, chatbot_display, existing_images],
        ).success(
            stream_model_output,
            [existing_images, chatbot, chatbot_display, max_gen_len, seed, gen_t, cfg, image_top_k, text_top_k],
            [chatbot, chatbot_display, existing_images],
        )
        submit_button.click(check_input_sanity, [text_input, new_images], []).success(
            show_user_input,
            [text_input, new_images, chatbot, chatbot_display, existing_images],
            [text_input, new_images, chatbot, chatbot_display, existing_images],
        ).success(
            stream_model_output,
            [existing_images, chatbot, chatbot_display, max_gen_len, seed, gen_t, cfg, image_top_k, text_top_k],
            [chatbot, chatbot_display, existing_images],
        )
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
