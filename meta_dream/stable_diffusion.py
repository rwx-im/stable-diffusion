import os
import torch
import io
import time


import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from rq import get_current_job

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


from upload import upload_image_data

initialized = False
model = None

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def init():
    global config
    global model
    global sampler

    config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, f"models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = PLMSSampler(model)

    os.makedirs("output", exist_ok=True)


def do_work(prompt):
    global initialized
    global model
    assert prompt is not None

    if not initialized:
        init()
        initialized = True

    job = get_current_job()

    batch_size = 1
    data = [batch_size * [prompt.text]]
    precision_scope = autocast
    n_samples = 1
    scale = 7.5
    C = 4
    H = 512
    W = 512
    f = 8
    ddim_eta = 0.0
    ddim_steps = 50
    start_code = None
    sample_path = "outputs"

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                print(f"Using seed: {prompt.seed}")
                seed_everything(prompt.seed)

                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )

                    output_path = os.path.join(sample_path, f"{job.id}.png")
                    buffer = io.BytesIO()

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        Image.fromarray(x_sample.astype(np.uint8)).save(buffer, "PNG")

                return upload_image_data(buffer, job.id, prompt)
