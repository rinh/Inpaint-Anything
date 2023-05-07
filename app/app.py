import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")

import base64
from io import BytesIO
import io
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder 
from pydantic import BaseModel, Field, create_model
import time 
from typing import Dict, List

import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
# from omegaconf import OmegaConf
# from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse

def setup_args(parser):
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="./models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./models/sam/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

 
def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    point_coords = [w, h]
    point_labels = [1]

    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w

    # model['sam'].set_image(img) # todo : update here for accelerating
    print(point_coords)
    masks, _, _ = model['sam'].predict(
        point_coords=np.array([point_coords]),
        point_labels=np.array(point_labels),
        multimask_output=True,
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    figs = []
    for idx, mask in enumerate(masks):
        # save the pointed and masked image
        tmp_p = mkstemp(".png")
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [point_coords], point_labels,
                    size=(width*0.04)**2)
        show_mask(plt.gca(), mask, random_color=False)
        plt.tight_layout()
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    return *figs, *masks


def get_inpainted_img(img, mask0, mask1, mask2):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for mask in [mask0, mask1, mask2]:
        if len(mask.shape)==3:
            mask = mask[:,:,0]
        img_inpainted = inpaint_img_with_builded_lama(
            model['lama'], img, mask, lama_config, device=device)
        out.append(img_inpainted)
    return out


# get args 
parser = argparse.ArgumentParser()
setup_args(parser)
args = parser.parse_args(sys.argv[1:])
# build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p=args.sam_ckpt
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks() as demo:
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row().style(mobile_collapse=False, equal_height=True):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image")
            with gr.Row():
                img = gr.Image(label="Input Image").style(height="200px")
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Pointed Image")
            with gr.Row():
                img_pointed = gr.Plot(label='Pointed Image')
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            with gr.Row():
                w = gr.Number(label="Point Coordinate W")
                h = gr.Number(label="Point Coordinate H")
            dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=100, step=1, value=15)
            sam_mask = gr.Button("Predict Mask", variant="primary").style(full_width=True, size="sm")
            lama = gr.Button("Inpaint Image", variant="primary").style(full_width=True, size="sm")
            clear_button_image = gr.Button(value="Reset", label="Reset", variant="secondary").style(full_width=True, size="sm")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Segmentation Mask")
            with gr.Row():
                mask_0 = gr.outputs.Image(type="numpy", label="Segmentation Mask 0").style(height="200px")
                mask_1 = gr.outputs.Image(type="numpy", label="Segmentation Mask 1").style(height="200px")
                mask_2 = gr.outputs.Image(type="numpy", label="Segmentation Mask 2").style(height="200px")

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image with Mask")
            with gr.Row():
                img_with_mask_0 = gr.Plot(label="Image with Segmentation Mask 0")
                img_with_mask_1 = gr.Plot(label="Image with Segmentation Mask 1")
                img_with_mask_2 = gr.Plot(label="Image with Segmentation Mask 2")

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Removed with Mask")
            with gr.Row():
                img_rm_with_mask_0 = gr.outputs.Image(
                    type="numpy", label="Image Removed with Segmentation Mask 0").style(height="200px")
                img_rm_with_mask_1 = gr.outputs.Image(
                    type="numpy", label="Image Removed with Segmentation Mask 1").style(height="200px")
                img_rm_with_mask_2 = gr.outputs.Image(
                    type="numpy", label="Image Removed with Segmentation Mask 2").style(height="200px")


    def get_select_coords(img, evt: gr.SelectData):
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        show_points(plt.gca(), [[evt.index[0], evt.index[1]]], [1],
                    size=(width*0.04)**2)
        return evt.index[0], evt.index[1], fig

    img.select(get_select_coords, [img], [w, h, img_pointed])
    img.upload(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])

    sam_mask.click(
        get_masked_img,
        [img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
        [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
    )

    lama.click(
        get_inpainted_img,
        [img, mask_0, mask_1, mask_2],
        [img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2]
    )


    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        [img, features, img_pointed, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2],
        [img, features, img_pointed, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2]
    )



def base64_to_img(base64_string):
    img_bytes = BytesIO(base64.b64decode(base64_string))
    img = Image.open(img_bytes)
    return img


def img_to_base64(img: Image,format='JPEG'):
    _img = img
    buffered = io.BytesIO()
    _img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str



class RmAnyRequest(BaseModel):
    image: str = Field(default=False, title="Image",
                       description="Image to work on, must be a Base64 string containing the image's data.")
    mask: str = Field(default=False, title="Mask Image",
                      description="Image to work on, must be a Base64 string containing the image's data.")
    shape: List[int] = Field(title="Images", description="List of images to work on. Must be Base64 strings")


class RmAnyResponse(BaseModel):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")



class Api:
    
    def __init__(self, app: FastAPI):
        self.router = APIRouter()
        self.app = app

        self.add_api_route("/api/v1/hi", self.hi_api, methods=["GET"]) 
        self.add_api_route("/api/v1/rm_any", self.rm_any_api, methods=["POST"], response_model=RmAnyResponse)

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)


    def hi_api(self):
        return "hi."


    def rm_any_api(self, req: RmAnyRequest):
        img = np.array(base64_to_img(req.image))
        mask = np.frombuffer(base64.b64decode(req.mask), dtype=np.uint8).reshape( req.shape )
        
        d_mask = dilate_mask(mask, 15)

        ia_path = Path(__file__).parent.absolute() / ".."
        result_img_arr = inpaint_img_with_lama(
            img,
            d_mask,
            config_p=str(ia_path / "./lama/configs/prediction/default.yaml"),
            ckpt_p=str(ia_path / "./models/big-lama"),
            device='cuda'
        )
        return RmAnyResponse(image=img_to_base64(Image.fromarray(result_img_arr)))




def create_api(app):
    api = Api(app) 
    return api



def wait_on_server():
    while 1:
        time.sleep(0.5)
        
        

if __name__ == "__main__":
    app, local_url, share_url = demo.launch(
            server_name="0.0.0.0",
            server_port=27766,
            prevent_thread_lock=True,
        )
    
    create_api(app)
    
    wait_on_server()
    