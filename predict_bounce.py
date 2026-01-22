import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import json

import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *
from bounce_detection import BounceDetector

def predict_batch(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    

def write_pred_video_with_events(video_file, pred_dict, events, save_file, traj_len=8, label_df=None):
    """ Write a video with prediction result and bounce events. """
    from collections import deque
    
    # Read video
    cap = cv2.VideoCapture(video_file)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(original_fps)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Read ground truth label if exists
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    
    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Organize events by frame
    events_by_frame = {}
    for event in events:
        frame_idx = int(event['frame'])
        events_by_frame[frame_idx] = event

    # Video config
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))
    
    # Create a queue for storing trajectory
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()
    
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Check if we have prediction for this frame
        current_x, current_y, current_vis = 0, 0, 0
        if i < len(x_pred):
            current_x, current_y, current_vis = x_pred[i], y_pred[i], vis_pred[i]
        
        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()
        
        # Push ball coordinates for each frame
        if label_df is not None:
             if i < len(label_df) and vis[i]:
                 gt_queue.appendleft([x[i], y[i]])
             else:
                 gt_queue.appendleft(None)
        
        if current_vis:
            pred_queue.appendleft([current_x, current_y])
        else:
            pred_queue.appendleft(None)

        # Draw ground truth trajectory if exists
        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        
        # Draw prediction trajectory
        frame = draw_traj(frame, pred_queue, color='yellow')
        
        # Draw events
        # We draw events that happened recently to make them visible for a few frames
        # or just draw strictly on the frame
        if i in events_by_frame:
            event = events_by_frame[i]
            e_type = event['event_type']
            color = (0, 0, 255) if e_type == 'landing' else (255, 0, 0) # Red for landing (BGR), Blue for hit
            if e_type == 'hit':
                color = (255, 0, 0)
            elif e_type == 'landing':
                color = (0, 0, 255)
            
            # Draw circle and text
            cv2.circle(frame, (int(event['x']), int(event['y'])), 10, color, -1)
            cv2.putText(frame, e_type, (int(event['x'])+15, int(event['y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        out.write(frame)
        i+=1

    out.release()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    parser.add_argument('--detect_bounce', action='store_true', default=True, help='whether to detect bounce events')
    args = parser.parse_args()

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    if '\\' in video_name:
         video_name = video_name.split('\\')[-1]

    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')
    out_events_file = os.path.join(args.save_dir, f'{video_name}_events.json')
    out_events_csv = os.path.join(args.save_dir, f'{video_name}_events.csv')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    cap = cv2.VideoCapture(args.video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    if args.eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode, 
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            print(f'Video length: {dataset.video_len}')
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # Predict
            tmp_pred = predict_batch(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
            
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:, :, :, ::-1])
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            video_len = len(frame_list)
        
        # Init prediction buffer params
        num_sample, sample_count = video_len-seq_len+1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, args.eval_mode)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0) / (sample_count+1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b+f, frame_i].sum(0) / (seq_len-f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = predict_batch(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if args.eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = predict_batch(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict_batch(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]
        

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)
    print(f'Prediction saved to {out_csv_file}')

    # Bounce Detection
    events = []
    if args.detect_bounce:
        print("Running Bounce Detection...")
        detector = BounceDetector(fps=int(fps))
        
        # Prepare data for detector
        x_raw = np.array(pred_dict['X'])
        y_raw = np.array(pred_dict['Y'])
        vis_raw = np.array(pred_dict['Visibility'])
        
        # Detector expects frames where visibility might be 0, but lists should match
        # The pred_dict has frames, x, y, vis
        # We need to make sure they are sorted by frame (which they should be)
        
        candidates = detector.detect(x_raw, y_raw, vis_raw)
        
        # Filter relevant events
        events = [c for c in candidates if c['event_type'] in ['landing', 'hit']]
        
        # Save events
        # JSON
        import json
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(out_events_file, 'w') as f:
            json.dump(events, f, cls=NpEncoder, indent=4)
        
        # CSV
        if events:
            events_df = pd.DataFrame(events)
            events_df.to_csv(out_events_csv, index=False)
        else:
            print("No events detected.")
            
        print(f'Events saved to {out_events_file}')
        
        # Print summary
        n_landing = sum(1 for e in events if e['event_type'] == 'landing')
        n_hit = sum(1 for e in events if e['event_type'] == 'hit')
        print(f"Detected {len(events)} events: {n_landing} landings, {n_hit} hits")

    # Write video with predicted coordinates AND events
    if args.output_video:
        print(f"Writing video to {out_video_file}...")
        write_pred_video_with_events(video_file, pred_dict, events, save_file=out_video_file, traj_len=args.traj_len)

    print('Done.')
