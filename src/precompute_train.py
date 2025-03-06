import argparse
import numpy as np
import pandas as pd
import random
import torch
import os
import time
import logging
import json

import torch.utils
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
from lavis.models import load_model_and_preprocess
from utility import base_path, device, params
import utility
from tqdm import tqdm
import copy
import data
from lavis.models.blip2_models.blip2_qformer_cir_image_diff_features import Blip2QformerCirImageDiffFeatures
from precompute_evaluation import evaluate_features

def set_seed(seed: int = 42, shuffle_seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(shuffle_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def blip_finetune(args):
    blip_model_name = 'blip2_cir_image_diff_features'
    if args.exp_name:
        training_path = base_path / 'log_TME' / f'{args.dataset}'/ args.exp_name
    else:    
        training_path = base_path / 'log_TME' / f'{args.dataset}'/ args.timestamp
    os.makedirs(training_path, exist_ok=True)
    backbone = args.backbone
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()
    preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
    
    dataset = data.get_dataset(args.dataset, preprocess, 'train', mode='relative', noise_ratio=args.noise_ratio)
    learning_rate = args.lr
    num_epochs = args.num_epochs
    loss_balance_dict = {
        'lrm': args.lrm, # it should be 1.0
        'lpm': args.lpm, 
        'lsa': args.lsa,
        'lrd': args.lrd,
    }
    
    optimizer = torch.optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':args.weight_decay}])
    
    optimizer_proj = torch.optim.AdamW([
            {'params': blip_model.d2t_proj.parameters(), 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': args.weight_decay},
            {'params': [blip_model.prompt_tokens], 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': args.weight_decay}
        ])
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, 
                           div_factor=100., steps_per_epoch=len(dataloader), epochs=num_epochs)
    
    scaler = torch.cuda.amp.GradScaler()

    training_log_frame = pd.DataFrame()
    accuracy_list = []
    best_acc = 0
    dataset.load_image_features()
    
    valset = data.get_dataset(args.dataset, preprocess, 'val', mode='relative')
    valset.load_image_features()
    
    for epoch in range(num_epochs):
        # print('curent_lr:', optimizer.param_groups[0]['lr'])
        # shuffle_seed_epoch = random.getstate()
        # torch.save(shuffle_seed_epoch, f'./RNG_States/shuffle_seed_epoch_{epoch}.pt')
        warmup = None
        pn_loss = copy.deepcopy(args.pn_loss)
        if args.method == 'image_diff':
            if epoch < args.warmup_qformer:
                warmup = 'qformer'
            elif epoch < args.warmup_qformer + args.warmup_proj:
                warmup = 'proj'
            elif epoch < args.warmup_qformer + args.warmup_proj + args.warmup_last:
                warmup = 'last'
        else:
            if epoch < args.warmup_epoch:
                warmup = 'wamrup'
        
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        partitioner_type = args.partitioner
        if warmup:
            partitioner_type = 'all_positive'
            pn_loss['positive_loss'] = pn_loss['warmup_loss']
            pn_loss['negative_loss'] = 'None'
            pn_loss['positive_align_loss'] = pn_loss['warmup_align_loss']
            pn_loss['negative_align_loss'] = 'None'
            pn_loss['trade_off'] = 1.0
            pn_loss['trade_off_align'] = 1.0
                   
        partitioner = utility.Partitioner(partitioner_type, args.split_type, args.threshold, 
                                          timestamp=args.timestamp, epoch=epoch, dataset_name=args.dataset)
        label_mask = partitioner.fit_features(blip_model, dataloader, txt_processors, debug=args.debug)
        label_mask = label_mask.to(device)
        
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(dataloader, ncols=120, mininterval=30)
        # losses_before = []
        # count = 0
        # debug = False
        for reference_name, target_hard_name, captions, index in train_bar:
            # time1 = time.
            reference_images = dataset.get_image_features(reference_name).to(device, non_blocking=True)
            target_images = dataset.get_image_features(target_hard_name).to(device, non_blocking=True)
            optimizer.zero_grad()
            # if count == 80:
            #     exit(0)
            labels = label_mask[index]
            if args.dataset == 'FashionIQ':
                flattened_captions = np.array(captions).T.flatten().tolist()
                captions = utility.generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors['eval'](caption) for caption in captions]
            blip_model.train()
            samples = {"image": reference_images, "target": target_images, "text_input":captions}
            with torch.cuda.amp.autocast():
                loss_dict = blip_model(samples, labels, pn_loss, warmup)
            loss = 0.
            for key in loss_dict:
                if key in loss_balance_dict:
                    loss += loss_balance_dict[key] * loss_dict[key]
                else:
                    raise ValueError('loss type is invalid')
            
            # count += 1
            scaler.scale(loss).backward()
            if warmup == 'proj':
                scaler.step(optimizer_proj)
            else:
                scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            images_in_batch = reference_images.shape[0]
            for key in loss_dict.keys():
                if key not in train_running_results:
                    train_running_results[key] = 0
                train_running_results[key] += loss_dict[key].to('cpu').detach().item() * images_in_batch
            
            train_running_results['images_in_epoch'] += images_in_batch
            bar_content = ''
            for key in train_running_results:
                if key != 'images_in_epoch':
                    bar_content += f"{key}: {train_running_results[key] / train_running_results['images_in_epoch']:.3f}, "
            train_bar.set_description(desc=f"[{epoch+1}/{num_epochs}] " f"{bar_content}", refresh=False)
            
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv((training_path / 'train_metrics.csv'), index=False)
        
        # evaluation
        blip_model.eval()
        # evaluate:
        accuracy_dict = evaluate_features(model=blip_model, dataset=valset, text_preprocessor=txt_processors["eval"])
        cur_acc = accuracy_dict['acc']
        if cur_acc > best_acc and args.save_training:
            best_acc = cur_acc
            logging.info('Save the current best model weights by mean average')
            torch.save(blip_model.state_dict(), training_path / 'best_model.pth')
        accuracy_list.append(accuracy_dict['acc'])
        
    with open('./res_acc.log', 'a+') as f:
        f.write(f'{args.timestamp}: {args.exp_name}\n')
        f.write(f'{str(accuracy_dict)}\n')
    if args.save_training:
        logging.info('Save the last epoch model')
        torch.save(blip_model.state_dict(), training_path / '30_epoch_model.pth')
    return accuracy_list

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='CIRR')
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--nc_type', type=str, default='mix')
    parser.add_argument('--method', type=str, default='image_diff')
    parser.add_argument('--debug', action='store_true', help='the debug of partitioner')
    # basic setting
    parser.add_argument('--backbone', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle_seed', type=int, default=42, help='The seed for shuffle data')
    parser.add_argument('--num_workers', type=int, default=9)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, help='The index of used gpu', default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)
    
    # method setting
    parser.add_argument('--positive_loss', type=str, default='RCL')
    parser.add_argument('--negative_loss', type=str, default='None')
    parser.add_argument('--positive_align_loss', type=str, default='MSE')
    parser.add_argument('--negative_align_loss', type=str, default='None')
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--trade_off_align', type=float, default=1.0)

    # loss weight
    parser.add_argument('--lrm', type=float, help='The loss weight of L_rm. It should be 1.0', default=1.0)
    parser.add_argument('--lpm', type=float, help='The loss weight of L_pm. It is gamma.', default=1.0)
    parser.add_argument('--lsa', type=float, help='The loss weight of L_sa. It is alpha.', default=1.0)
    parser.add_argument('--lrd', type=float, help='The loss weith of L_rd. It is beta.', default=0.2)
    
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--warmup_qformer', type=int, default=3)
    parser.add_argument('--warmup_proj', type=int, default=2)
    parser.add_argument('--warmup_last', type=int, default=1)
    parser.add_argument('--warmup_loss', type=str, default='RCL')
    parser.add_argument('--warmup_align_loss', type=str, default='MSE')
    
    # partitioner
    parser.add_argument('--partitioner', type=str, default='GMM')
    parser.add_argument('--split_type', type=str, default='loss')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_training', action='store_true', help='save model in training.')
    
    # experiment_name
    parser.add_argument('--exp_name', type=str, default='')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    utility.set_device(args.gpu)
    set_seed(args.seed, args.shuffle_seed)
    if args.dataset.lower() == 'cirr':
        args.dataset = 'CIRR'
    elif args.dataset.lower() == 'fashioniq':
        args.dataset = 'FashionIQ'
    else:
        raise ValueError(f'The name of dataset {args.dataset} is invalid.')
    
    log_folder_path, timestamp = utility.get_log(args.dataset, args.exp_name)
    args.timestamp = timestamp
    file_name = './log_TME/parameters.json'
    os.makedirs('./log_TME', exist_ok=True)
    utility.Params.initialize(args)
    logging.info('Arguments:')
    for k in args.__dict__.keys():
        logging.info(f'    {k}:, {str(args.__dict__[k])}')
    accuracy_list = blip_finetune(params)
    this_dict = {timestamp:{'parameters': params(), 'accuracies': [round(num, 2) for num in accuracy_list], 
                            'max_acc':round(max(accuracy_list), 2), 'max_acc_epoch': int(np.argmax(accuracy_list)+1),
                            'last_epoch_acc':round(accuracy_list[-1], 2),
                            }}
    if os.path.exists(file_name):
        with open(file_name, 'r') as json_file: 
            my_dict = json.load(json_file)
        my_dict.update(this_dict)
    else:
        my_dict = this_dict

    formatted_json_string = utility.custom_json_dumps(my_dict, indent=2)
    with open(file_name, 'w') as f:
        f.write(formatted_json_string)