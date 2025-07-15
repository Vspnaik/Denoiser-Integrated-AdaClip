import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch
from scipy.ndimage import gaussian_filter
import cv2

# Importing from local modules
from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method import AdaCLIP_Trainer
from PIL import Image
import numpy as np

from UNet.unet import UNet

setup_seed(111)

def train(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."

    # Configurations
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_fig = args.save_fig

    # Logger
    logger = Logger('log.txt')

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # Prepare model
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)

    model.load(args.ckt_path)

    if args.testing_model == 'dataset':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
                                                         f"{dataset_dict.keys()}"

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        os.makedirs(csv_root, exist_ok=True)  # Create directory if it doesn't exist
        os.makedirs(image_root, exist_ok=True)  # Create directory if it doesn't exist
        csv_path = os.path.join(csv_root, f'{args.testing_data}.csv')
        image_dir = os.path.join(image_root, f'{args.testing_data}')
        os.makedirs(image_dir, exist_ok=True)

        test_data_cls_names, test_data, test_data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_fig_flag = save_fig

        metric_dict = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig_flag,
            image_dir,
        )

        for tag, data in metric_dict.items():
            logger.info(
                '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.
                    format(tag,
                           data['auroc_im'],
                           data['f1_im'],
                           data['ap_im'],
                           data['auroc_px'],
                           data['f1_px'],
                           data['ap_px'])
            )

        for k in metric_dict.keys():
            write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    elif args.testing_model == 'image':
        assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"
        
        # Create save directory if it doesn't exist
        os.makedirs(args.save_path, exist_ok=True)
        
        ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil_img = Image.open(args.image_path).convert('RGB')

        img_input = model.preprocess(pil_img).unsqueeze(0)
        img_input = img_input.to(model.device)
        

        try:
            unet = UNet(3).to(model.device)
            model_path = args.unet_path if hasattr(args, 'unet_path') else './denoisers/Trained_Weights/unet_epoch_10.pth'
            
            if os.path.isfile(model_path):
                unet.load_state_dict(torch.load(model_path, map_location=model.device))
                unet.eval()
                img_input = unet(img_input)
                logger.info(f"Successfully loaded and applied UNet from {model_path}")
            else:
                logger.info(f"UNet model not found at {model_path}, proceeding without denoising")
        except Exception as e:
            logger.info(f"Error loading UNet: {str(e)}, proceeding without denoising")

        with torch.no_grad():
            anomaly_map, anomaly_score, patch_tokens = model.clip_model(img_input, [args.class_name], aggregation=True)

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        
        # Normalize anomaly map to 0-255 range for visualization
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_map_norm = (anomaly_map_norm * 255).astype(np.uint8)
        heat_map = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
        vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)

        vis_map = cv2.hconcat([ori_image, vis_map])
        save_path = os.path.join(args.save_path, args.save_name)
        
        # Save the result
        cv2.imwrite(save_path, vis_map)
        
        # Explicitly verify the file was created
        if os.path.isfile(save_path):
            logger.info(f"Anomaly detection results saved to {save_path}, with anomaly score: {anomaly_score:.3f}")
            print(f"Anomaly detection results saved to {save_path}, with anomaly score: {anomaly_score:.3f}")
        else:
            error_msg = f"Failed to save results to {save_path}. Check directory permissions."
            logger.info(error_msg)
            print(error_msg)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_mvtec_colondb.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec_colondb.pth)")

    parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")

    # for the dataset model
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")

    # for the image model
    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Model for testing (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="candle",
                        help="The class name of the testing image (default: 'candle')")
    parser.add_argument("--save_name", type=str, default="test.png",
                        help="Model for testing (default: 'dataset')")

    parser.add_argument("--unet_path", type=str, default="./denoisers/Trained_Weights/unet_epoch_3.pth",
                        help="Path to the UNet denoiser model (default: './denoisers/Trained_Weights/unet_epoch_3.pth')")

    parser.add_argument("--save_path", type=str, default='./workspaces',
                        help="Directory to save results (default: './workspaces')")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")

    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations (default: False)")

    # Hyper-parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")

    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    train(args)