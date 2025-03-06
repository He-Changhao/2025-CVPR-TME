import argparse
import utility
from precompute_evaluation import evaluate_features
from utility import device, base_path
from lavis.models import load_model_and_preprocess
import torch
from tqdm import tqdm
from operator import itemgetter
import numpy as np
import json
import os
from pathlib import Path
import logging
import data
import sys

def cirr_test(model, dataset, text_preprocessor, num_workers, folder):
    dataloader = torch.utils.data.DataLoader(dataset, 32, num_workers=num_workers, shuffle=False, pin_memory=True)
    Qformer_bs = 256
    n_images = dataset.images.shape[0]
    F_images = []
    logging.info('compute image last hidden states')
    for idx in tqdm(range(0, n_images, Qformer_bs), ncols=150):
        cur_image_embed = dataset.images[idx:idx+Qformer_bs].to(device)
        with torch.no_grad():
            F_image = model.encode_image(cur_image_embed)
        F_images.append(F_image)
    F_images = torch.cat(F_images, dim=0)
    index_names = list(dataset.name_to_relpath.keys())
    reference_names = []
    distances = []
    subset_names = []
    print('Compute simlarity: ')
    pairids = []
    for pair_id, reference_name, captions, group_members in tqdm(dataloader, ncols=150):
        r_indices = [dataset.name_to_idx[name] for name in reference_name]
        F_r = F_images[r_indices].to(device)
        captions = [text_preprocessor(item) for item in captions]
        group_members = np.array(group_members).T.tolist()
        distance = model.inference(F_r, F_images, captions)
        distances.append(distance)

        pairids.extend(pair_id.tolist())
        reference_names.extend(reference_name)
        subset_names.extend(group_members)
    
    distances = torch.vstack(distances)
    distances = 1 - distances
    ref_images_indices_in_index_names = []
    for i in reference_names:
            ref_images_indices_in_index_names.append(index_names.index(i))
    distances[list(range(distances.shape[0])), ref_images_indices_in_index_names] = 10e10
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    subset_names = np.array(subset_names)
    group_mask = (sorted_index_names[..., None] == subset_names[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairids, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairids, sorted_group_names)}
    
    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    print(f"Saving CIRR test predictions")
    with open(folder / f"recall_submission_.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(folder / f"recall_subset_submission_.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIRR', help='CIRR or FashionIQ')
    parser.add_argument('--mode', type=str, default='test', help='test of validate')
    parser.add_argument('--model-path', type=str, default='models_weight/cirr-0-best.pth', help='the path of model')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu device used for test')
    parser.add_argument('--name', type=str, default='', help='the save folder name')
    parser.add_argument('--num_workers', type=int, default=9)
    args = parser.parse_args()
    return args
    
def main():
    # logs setting
    log_format = '%(asctime)s: %(message)s'
    date_format = '%Y-%m-%d-%H-%M-%S'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format, datefmt=date_format)
    
    args = parser_args()
    utility.set_device(args.gpu)
    if args.dataset.lower() == 'cirr':
        args.dataset = 'CIRR'
    elif args.dataset.lower() == 'fashioniq':
        args.dataset = 'FashionIQ'
    else:
        raise ValueError(f'The name of dataset {args.dataset} is invalid.')

    if args.mode == 'test' and args.dataset != 'CIRR':
        raise ValueError('FashionIQ has no test set.')
    preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
    if args.dataset not in ['CIRR', 'FashionIQ']:
        raise ValueError('Dataset name is invalid.')
    
    model_name = 'blip2_cir_image_diff_features'
    backbone = 'pretrain'
    blip_model, _, txt_processors = load_model_and_preprocess(name=model_name, model_type=backbone, is_eval=False, device=device)
    
    folder = os.path.dirname(args.model_path)
    model_path = base_path / args.model_path
    save_folder = base_path / folder
    if args.name:
        save_folder = save_folder / args.name
        
    log_path = os.path.join(save_folder, 'process.log')
    utility.get_log_simple(log_path)
    logging.info('Arguments:')
    for k in args.__dict__.keys():
        logging.info(f'    {k}:, {str(args.__dict__[k])}')
        
    checkpoint = torch.load(model_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint, strict=False)
    print("Missing keys {}".format(msg.missing_keys))
    blip_model.eval()
    if args.mode == 'validate':
        dataset = data.get_dataset(args.dataset, preprocess, 'val')
        dataset.load_image_features()
        acc_dict = evaluate_features(blip_model, dataset, txt_processors['eval'])
        print(acc_dict)
    elif args.mode == 'test':
        dataset = data.get_dataset('CIRR', preprocess, 'test1')
        dataset.load_image_features()
        cirr_test(blip_model, dataset, txt_processors['eval'], args.num_workers, folder=save_folder)
        print('Test submission has been generated.')
    else:
        raise ValueError('mode is invalid')
    
if __name__ == '__main__':
    main()