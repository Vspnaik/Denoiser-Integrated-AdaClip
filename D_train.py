from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import torch
from scipy.ndimage import gaussian_filter
import cv2
import torch.nn as nn
import torch.optim as optim

# Importing from local modules
from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method import AdaCLIP_Trainer
from PIL import Image
import numpy as np
import pandas as pd


# importing denoisers
from denoisers.denoiser import TransformerEncoder

setup_seed(111)
def load_data():
    csv_path = '/data/rrjha/AP/AdaCLIP_Denoiser/preprocess_training_datasets/mvtec_image_paths.csv'
    df = pd.read_csv(csv_path)
    df1 = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df1

def add_gaussian_noise(img, std_dev):
    """Adds Gaussian noise to a PIL image."""
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std_dev, img_np.shape)
    noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

from torch.utils.data import Dataset, DataLoader
# Custom dataset class
import torch.nn.functional as F

def cosine_loss(x1, x2):
    return 1 - F.cosine_similarity(x1, x2, dim=-1).mean()
class FeaturePairDataset(Dataset):
    def __init__(self, noisy_features, normal_feature):
        # noisy_features: list of 10 tensors
        # normal_feature: single tensor (will be repeated)
        self.noisy_features = noisy_features
        self.normal_feature = normal_feature

    def __len__(self):
        return len(self.noisy_features)

    def __getitem__(self, idx):
        return self.noisy_features[idx], self.normal_feature
    
def train(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."

    # Configurations
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save_fig = args.save_fig

    # Logger
    # logger = Logger('log.txt')

    # Print basic information
    # for key, value in sorted(vars(args).items()):
    #     logger.info(f'{key} = {value}')


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

    # if args.testing_model == 'dataset':
    #     assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
    #                                                      f"{dataset_dict.keys()}"

    #     save_root = args.save_path
    #     csv_root = os.path.join(save_root, 'csvs')
    #     image_root = os.path.join(save_root, 'images')
    #     csv_path = os.path.join(csv_root, f'{args.testing_data}.csv')
    #     image_dir = os.path.join(image_root, f'{args.testing_data}')
    #     os.makedirs(image_dir, exist_ok=True)

    #     test_data_cls_names, test_data, test_data_root = get_data(
    #         dataset_type_list=args.testing_data,
    #         transform=model.preprocess,
    #         target_transform=model.transform,
    #         training=False)

    #     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #     save_fig_flag = save_fig

    #     metric_dict = model.evaluation(
    #         test_dataloader,
    #         test_data_cls_names,
    #         save_fig_flag,
    #         image_dir,
    #     )

    #     for tag, data in metric_dict.items():
    #         logger.info(
    #             '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.
    #                 format(tag,
    #                        data['auroc_im'],
    #                        data['f1_im'],
    #                        data['ap_im'],
    #                        data['auroc_px'],
    #                        data['f1_px'],
    #                        data['ap_px'])
    #         )


    #     for k in metric_dict.keys():
    #         write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    # elif args.testing_model == 'image':
    df = load_data()

    # Initialize Denoisers
    D1 = TransformerEncoder(1024, 4, 16).to(device)
    D2 = TransformerEncoder(1024, 4, 16).to(device)
    D3 = TransformerEncoder(1024, 4, 16).to(device)
    D4 = TransformerEncoder(1024, 4, 16).to(device)
    
    criterion = nn.CosineEmbeddingLoss()

    denoisers = [D1, D2, D3, D4]
    weights_dir = '/data/rrjha/AP/AdaCLIP_Denoiser/denoisers/Trained_Weights'

    optimizers = [AdamW(d.parameters(), lr=0.00001, weight_decay=0.01) for d in denoisers]
    error = {1:[],2:[],3:[],4:[]}
    with tqdm(total=len(df), desc="Processing Images", unit="image") as pbar:
    # Track cumulative loss for each denoiser
        denoiser_losses = {f'D{i}': 0.0 for i in range(1, 5)}
        for index, row in df.iterrows():
            image_path = row['image_path']
            class_label = row['class']
            assert os.path.isfile(image_path), f"Please verify the input image path: {image_path}"

            pil_img = Image.open(image_path).convert('RGB')

            img_input = model.preprocess(pil_img).unsqueeze(0)
            img_input = img_input.to(model.device)
            nfeature1 = None
            nfeature2 = None
            nfeature3 = None
            nfeature4 = None
            load_path = "/data/rrjha/AP/AdaCLIP_Denoiser/train_data/patch_tokens.pt"
            
            with torch.no_grad():
                anomaly_map, anomaly_score, patch_tokens = model.clip_model(img_input, [class_label], aggregation=True)
                
                patch_tokens = torch.load(load_path)
                os.remove(load_path)
                
                nfeature1 = patch_tokens[0]
                nfeature2 = patch_tokens[1]
                nfeature3 = patch_tokens[2]
                nfeature4 = patch_tokens[3]
            
            feature1 = []
            feature2 = []
            feature3 = []
            feature4 = []
            
            for i in range(10):
                std_dev = i * 2
                noisy_img = add_gaussian_noise(pil_img, std_dev)
                img_input = model.preprocess(noisy_img).unsqueeze(0)
                img_input = img_input.to(model.device)
                
                with torch.no_grad():
                    anomaly_map, anomaly_score, patch_tokens = model.clip_model(img_input, [class_label], aggregation=True)
                    
                    patch_tokens = torch.load(load_path)
                    os.remove(load_path)

                    feature1.append(patch_tokens[0])
                    feature2.append(patch_tokens[1])
                    feature3.append(patch_tokens[2])
                    feature4.append(patch_tokens[3])

            # Create datasets for each feature level
            dataset1 = FeaturePairDataset(feature1, nfeature1)
            dataset2 = FeaturePairDataset(feature2, nfeature2)
            dataset3 = FeaturePairDataset(feature3, nfeature3)
            dataset4 = FeaturePairDataset(feature4, nfeature4)

            # Create DataLoaders
            loader1 = DataLoader(dataset1, batch_size=1, shuffle=True)
            loader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
            loader3 = DataLoader(dataset3, batch_size=1, shuffle=True)
            loader4 = DataLoader(dataset4, batch_size=1, shuffle=True)

            loaders = [loader1, loader2, loader3, loader4]
            
            # Train each denoiser
            for i, (denoiser, loader, optimizer) in enumerate(zip(denoisers, loaders, optimizers), start=1):
                denoiser.train()
                running_loss = 0.0
                # Process each batch in the loader
                for noisy_feat, clean_feat in loader:
                    noisy_feat = noisy_feat.squeeze(0).to(device)      # shape: [1, 1024]
                    clean_feat = clean_feat.squeeze(0).to(device)      # shape: [1, 1024]
                    
                    optimizer.zero_grad()
                    output = denoiser(noisy_feat)
                    # loss = criterion(output, clean_feat)
                    loss = cosine_loss(output, clean_feat)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                
                # Update the cumulative loss for the denoiser
                error[i].append(running_loss)
                denoiser_losses[f'D{i}'] += running_loss
                loss_info = ', '.join([f"{key}:{denoiser_losses[key] / (pbar.n + 1):.6f}" for key in denoiser_losses])
                pbar.set_postfix({"Loss": loss_info})
                
            # Update progress bar after processing each row
            pbar.update(1)
            for i in range(4):
                model_save_path = os.path.join(weights_dir, f'denoiser{i+1}.pth')
                torch.save(denoisers[i].state_dict(), model_save_path)

            # Save error tensor
            error_path = os.path.join(weights_dir, 'Error.pth')
            torch.save(error, error_path)
    # anomaly_map = anomaly_map[0, :, :]
    # anomaly_score = anomaly_score[0]
    # anomaly_map = anomaly_map.cpu().numpy()
    # anomaly_score = anomaly_score.cpu().numpy()

    # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    # anomaly_map = anomaly_map * 255
    # anomaly_map = anomaly_map.astype(np.uint8)

    # heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    # vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)

    # vis_map = cv2.hconcat([ori_image, vis_map])
    # save_path = os.path.join(args.save_path, args.save_name)
    # print(f"Anomaly detection results are saved in {save_path}, with an anomaly of {anomaly_score:.3f} ")
    # cv2.imwrite(save_path, vis_map)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_mvtec_colondb.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec_colondb.pth)")

    # parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"],
    #                     help="Model for testing (default: 'dataset')")

    # for the dataset model
    # parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")

    # for the image model
    parser.add_argument("--image_path", type=str, default="asset/image.jpg",
                        help="Model for testing (default: 'asset/image.jpg')")
    # parser.add_argument("--class_name", type=str, default="candle",
    #                     help="The class name of the testing image (default: 'candle')")
    # parser.add_argument("--save_name", type=str, default="test.png",
    #                     help="Model for testing (default: 'dataset')")


    # parser.add_argument("--save_path", type=str, default='./workspaces',
    #                     help="Directory to save results (default: './workspaces')")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")

    # parser.add_argument("--save_fig", type=str2bool, default=False,
    #                     help="Save figures for visualizations (default: False)")

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