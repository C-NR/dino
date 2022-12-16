# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import glob
import sys
import argparse
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits


FOURCC = {
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
}
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoGenerator:
    def __init__(self, args):
        self.args = args
        # self.model = None
        # Don't need to load model if you only want a video
        if not self.args.video_only:
            self.model = self.__load_model()

    def run(self):
        if self.args.input_path is None:
            print(f"Provided input path {self.args.input_path} is non valid.")
            sys.exit(1)
        else:
            if self.args.video_only:
                attention_folder = os.path.join(
                    self.args.output_path, "attention"
                )
                self._generate_video_from_images(
                    attention_folder, self.args.output_path
                )
                # self._generate_video_from_images(
                #     self.args.input_path, self.args.output_path
                # )
            else:
                # If input path exists
                if os.path.exists(self.args.input_path):
                    # If input is a video file
                    if os.path.isfile(self.args.input_path):
                        frames_folder = os.path.join(self.args.output_path, "frames")
                        attention_folder = os.path.join(
                            self.args.output_path, "attention"
                        )

                        os.makedirs(frames_folder, exist_ok=True)
                        os.makedirs(attention_folder, exist_ok=True)

                        if not self.args.no_extract:
                            self._extract_frames_from_video(
                                self.args.input_path, frames_folder
                            )

                        self._inference(
                            frames_folder,
                            attention_folder,
                        )

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )

                    # If input is a folder of already extracted frames
                    if os.path.isdir(self.args.input_path):
                        attention_folder = os.path.join(
                            self.args.output_path, "attention"
                        )

                        os.makedirs(attention_folder, exist_ok=True)

                        self._inference(self.args.input_path, attention_folder)

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )

                # If input path doesn't exists
                else:
                    print(f"Provided input path {self.args.input_path} doesn't exists.")
                    sys.exit(1)

    def _extract_frames_from_video(self, inp: str, out: str):
        vidcap = cv2.VideoCapture(inp)
        self.args.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {inp} ({self.args.fps} fps)")
        print(f"Extracting frames to {out}")

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(out, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def _generate_video_from_images(self, inp: str, out: str):

        subfolder_list = sorted(glob.glob(os.path.join(out, "attention*")))
        # print(subfolder_list)
        for inp in subfolder_list:
            parts = inp.split('/')
            sub = parts[-1]
 
            files = glob.glob(os.path.join(inp, "attn-*.jpg"))
            files.extend(glob.glob(os.path.join(inp, "attn-*.png")))

            # attention_images_list = sorted(glob.glob(os.path.join(inp, "attn-*.jpg")))
            attention_images_list = sorted(files)
            # print(inp)
            # print(len(attention_images_list))
            # continue
            if len(attention_images_list) <= 0:
                continue
            size = None
            # thresh = 0
            # if self.args.threshold > 1 and self.args.blend:
            #     thresh = self.args.threshold

            for filename in tqdm(attention_images_list):
                # img = cv2.imread(filename,0)
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if size == None:
                    height, width, _ = img.shape
                    size = (width, height)
                    print(f"Generating video {size} to {out}")
                    break

            if size == None:
                print('Could not determine the image size for ' + inp)
            else: 
                vout = cv2.VideoWriter(
                    os.path.join(out, sub + "_video." + self.args.video_format),
                    FOURCC[self.args.video_format],
                    self.args.fps,
                    size,
                )

                for filename in tqdm(attention_images_list):
                    # img = cv2.imread(filename,0)
                    img = cv2.imread(filename)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if self.args.blend:
                        # print(filename.replace(sub + "/attn-","frames/"))
                        org = cv2.imread(filename.replace(sub + "/attn-","frames/"))
                        org = cv2.resize(org, size)
                        org = cv2.cvtColor(org, cv2.COLOR_RGB2BGR)
                        # img = cv2.blur(img, (20, 20))
                        # _, img = cv2.threshold(img,thresh,255,cv2.THRESH_TOZERO)
                        # _, mask = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
                        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        alpha = 0.5
                        beta = (1.0 - alpha)
                        img = cv2.addWeighted(img, alpha, org, beta, 0.0)
                        # img = cv2.bitwise_and(img, mask)
                    vout.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                vout.release()
                print("Done")

    def _inference(self, inp: str, out: str):
        print(f"Generating attention images to {out}")

        files = glob.glob(os.path.join(inp, "*.jpg"))
        files.extend(glob.glob(os.path.join(inp, "*.png")))

        # for img_path in tqdm(sorted(glob.glob(os.path.join(inp, "*.jpg")))):
        # for img_path in tqdm(sorted(glob.glob(os.path.join(inp, "*.png")))):
        for img_path in tqdm(sorted(files)):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

            if self.args.resize is not None:
                transform = pth_transforms.Compose(
                    [
                        pth_transforms.ToTensor(),
                        pth_transforms.Resize(self.args.resize),
                        pth_transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )
            else:
                transform = pth_transforms.Compose(
                    [
                        pth_transforms.ToTensor(),
                        pth_transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )

            img = transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.args.patch_size,
                img.shape[2] - img.shape[2] % self.args.patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // self.args.patch_size
            h_featmap = img.shape[-1] // self.args.patch_size

            attentions = self.model.get_last_selfattention(img.to(DEVICE))

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            # print(attentions)

            #exit(0)

            # save attentions heatmaps
            fname = os.path.join(out, "attn-" + os.path.basename(img_path))
            plt.imsave(
                fname=fname,
                arr=sum(
                    attentions[i] * 1 / attentions.shape[0]
                    for i in range(attentions.shape[0])
                ),
                cmap="hot",
                # cmap="gray",
                format="jpg",
            )

            for i in range(attentions.shape[0]):
                if not os.path.isdir(out + str(i)):
                    os.mkdir(out + str(i))
                fname = os.path.join(out + str(i), "attn-" + os.path.basename(img_path))
                plt.imsave(
                    fname=fname,
                    arr = attentions[i] * 1,
                    cmap="hot",
                    # cmap="gray",
                    format="jpg",
                )



    def __load_model(self):
        # build model
        model = vits.__dict__[self.args.arch](
            patch_size=self.args.patch_size, num_classes=0
        )
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)

        if os.path.isfile(self.args.pretrained_weights):
            state_dict = torch.load(self.args.pretrained_weights, map_location="cpu")
            if (
                self.args.checkpoint_key is not None
                and self.args.checkpoint_key in state_dict
            ):
                print(
                    f"Take key {self.args.checkpoint_key} in provided checkpoint dict"
                )
                state_dict = state_dict[self.args.checkpoint_key]
            print(state_dict.items())
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    self.args.pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            url = None
            if self.args.arch == "vit_small" and self.args.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif self.args.arch == "vit_small" and self.args.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif self.args.arch == "vit_base" and self.args.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif self.args.arch == "vit_base" and self.args.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print(
                    "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/dino/" + url
                )
                model.load_state_dict(state_dict, strict=True)
            else:
                print(
                    "There is no reference weights available for this model => We use random weights."
                )
        return model

def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out

def parse_args():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="""Path to a video file if you want to extract frames
            or to a folder of images already extracted by yourself.
            or to a folder of attention images.""",
    )
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        help="""Path to store a folder of frames and / or a folder of attention images.
            and / or a final video. Default to current directory.""",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or W H): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="""Use this flag if you only want to generate a video and not all attention images.
            If used, --input_path must be set to the folder of attention images. Ex: ./attention/""",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="""Use this flag if you want to generate mask tho original video with the results.""",
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="""Use this flag if you only want to generate a video and not extract all images.""",
    )
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="FPS of input / output video. Automatically set if you extract frames from a video.",
    )
    parser.add_argument(
        "--video_format",
        default="mp4",
        type=str,
        choices=["mp4", "avi"],
        help="Format of generated video (mp4 or avi).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vg = VideoGenerator(args)
    vg.run()
