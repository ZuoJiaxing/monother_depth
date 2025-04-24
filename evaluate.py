# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np
import json
import os

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.data.data_mono_thermal import ThermalDepthDataLoader
from zoedepth.data.data_thermal_rgbtd import RGBTDepthDataLoader
from zoedepth.data.data_mono_km_mix import MixedMS2KITTI
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics, compute_metrics_weighted, count_parameters)


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    #pred2 = model(torch.flip(images, [3]), **kwargs)
    #pred2 = get_depth_from_prediction(pred2)
    #pred2 = torch.flip(pred2, [3])

    #mean_pred = 0.5 * (pred1 + pred2)
    mean_pred = pred1

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    weighted_metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), config.min_depth_eval, config.max_depth_eval, cmap='jet')
            p = colorize(pred.squeeze().cpu().numpy(), config.min_depth_eval, config.max_depth_eval, cmap='jet')
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))



        # print(depth.shape, pred.shape)
        metrics_result = compute_metrics(depth, pred, config=config)
        if np.isnan(metrics_result['abs_rel']) or np.isnan(metrics_result['a1']) : # np.isnan(metrics_result['silog'])
            print("============= Warning, the metrics contains Nan! ===================")
            print(sample['image_path'])
            print("Predicted depth contains Nan: ", torch.isnan(pred).any())
            print(metrics_result)
        else:
            metrics.update(metrics_result)

        weighted_metrics_result = compute_metrics_weighted(depth, pred, config=config)
        if np.isnan(weighted_metrics_result['abs_rel']) or np.isnan(weighted_metrics_result['a1']):
            print("============= Warning, the weighted metrics contains Nan! ===================")
            print(sample['image_path'])
            print(weighted_metrics_result)
            print("Predicted depth contains Nan: ", torch.isnan(pred).any())
        else:
            weighted_metrics.update(weighted_metrics_result)

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    weighted_metrics = {k: r(v) for k, v in weighted_metrics.get_value().items()}
    return metrics, weighted_metrics

def main(config, mode='test'):
    model = build_model(config)
    if config.dataloader_type == 0:
        test_loader = DepthDataLoader(config, 'online_eval').data
    elif config.dataloader_type == 1:
        test_loader = ThermalDepthDataLoader(config, mode, modality=config.modality).data
    elif config.dataloader_type == 2:
        test_loader = RGBTDepthDataLoader(config, mode, modality=config.modality).data
    elif config.dataloader_type==3: # Mixed KITTI and MS2 RGB dataset
        train_loader = MixedMS2KITTI(config, "train", modality=config.modality).data
        test_loader = MixedMS2KITTI(config, "online_eval", modality=config.modality).data
    else:
        raise ValueError("Invalid dataloader_type")
    model = model.cuda()
    metrics, metrics_weighted = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(" | ".join([f"{metric}" for metric, value in metrics.items()]))
    print(" | ".join([f"{value:.3f}" for metric, value in metrics.items()]))
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"

    print(f"{colors.fg.yellow}")
    print(metrics_weighted)
    print(" | ".join([f"{metric}" for metric, value in metrics_weighted.items()]))
    print(" | ".join([f"{value:.3f}" for metric, value in metrics_weighted.items()]))
    print(f"{colors.reset}")
    metrics_weighted['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"

    ### Write the metrics in a txt file.
    if "save_images" in config and config.save_images:
        result_savepath = os.path.join(config.save_images, "evaluation_results.txt")
    else:
        result_savepath = "evaluation_results.txt"

    metrics_str = json.dumps(metrics)
    metrics_weighted_str = json.dumps(metrics_weighted)
    with open(result_savepath, 'a') as file:
        file.write('metrics: \n')
        file.write(metrics_str + '\n')
        file.write('metrics_weighted: \n')
        file.write(metrics_weighted_str + '\n')

    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', mode='test', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}, mode {mode}...")
    metrics = main(config, mode=mode)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")
    parser.add_argument("--testmode", type=str, required=True,
                        default='test')

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, mode=args.testmode, **overwrite_kwargs)
