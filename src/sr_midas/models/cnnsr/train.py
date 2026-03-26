"""CNN super-resolution model training.

Refactored from: SRML-HEDM/CNNSR/train.py
"""

import os
import sys
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sr_midas.models.cnnsr.dataset import trainData_CNNSR
from sr_midas.models.cnnsr.architecture import CNNSR

SEP = os.sep


# ----------------------------
def train_cnnsr(args):
    """Train a CNNSR super-resolution model.
    Args:
        args: namespace or dict with keys:
            expName (str): experiment name (used as output directory name)
            pst (str): path to .h5 patchstore file
            inSRx (int): input patch SR factor
            outSRx (int): output patch SR factor
            useRch (str): include Radius channel ("true"/"false")
            useEtach (str): include Eta channel ("true"/"false")
            arch (str): CNN architecture string e.g. "64-5-r_32-5-r_1-5-s"
            lr (float): learning rate
            lossF (str): loss function ("mse" or "mae")
            mbsz (int): mini batch size
            maxItr (int): max training iterations
            trainFrac (float): training dataset fraction
            nwork (int): number of DataLoader workers
            ecVal (float): error convergence value
            ecItr (int): convergence check interval (iterations)
            inPstPath (str or None): input patchstore path (if separate)
            outPstPath (str or None): target patchstore path (if separate)
            loadChkpt (str or None): path to pre-trained checkpoint for initialization
            trainOutDir (str): output directory for trained models
    """

    if isinstance(args, dict):
        import argparse
        args = argparse.Namespace(**args)

    # Extract architecture parameters
    l_layer_params = args.arch.split("_")
    l_ch_nrs = [int(i.split("-")[0]) for i in l_layer_params]
    l_ker_size = [int(i.split("-")[1]) for i in l_layer_params]
    l_act_func = [str(i.split("-")[2]) for i in l_layer_params]

    train_out_dir = f"{args.trainOutDir}{SEP}"
    if not os.path.isdir(train_out_dir):
        os.mkdir(train_out_dir)

    mod_dir = f"{train_out_dir}{args.expName}-itrOut{SEP}"
    if not os.path.isdir(mod_dir):
        os.mkdir(mod_dir)

    args_dict_savepath = f"{mod_dir}_train_args.json"
    with open(args_dict_savepath, "w") as f:
        json.dump(vars(args), f)

    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    logFilePath = f"{mod_dir}_train_log.log"
    logging.basicConfig(filename=logFilePath, encoding="utf-8", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("---------------------------------")
    logging.info(f"Device selected for training: {torch_devs}\n")
    logging.info("Loading data into device memory...")

    ds_train = trainData_CNNSR(pst_path=args.pst,
                               srfac_out=args.outSRx, srfac_in=args.inSRx,
                               use_R_channel=args.useRch, use_Eta_channel=args.useEtach,
                               normalize_R_channel=True, normalize_Eta_channel=True,
                               use="train", train_frac=args.trainFrac,
                               pst_path_X=args.inPstPath, pst_path_Y=args.outPstPath)

    ds_valid = trainData_CNNSR(pst_path=args.pst,
                               srfac_out=args.outSRx, srfac_in=args.inSRx,
                               use_R_channel=args.useRch, use_Eta_channel=args.useEtach,
                               normalize_R_channel=True, normalize_Eta_channel=True,
                               use="test", train_frac=args.trainFrac,
                               pst_path_X=args.inPstPath, pst_path_Y=args.outPstPath)

    logging.info("DONE.\n")
    logging.info(f"X shape: {ds_train.X.shape};   Y shape: {ds_train.Y.shape}")

    b_trainContinue = False
    modInitCount = 0

    while b_trainContinue == False:
        modInitCount += 1
        logging.info(f"\n---------Initiating MODEL: Count ({modInitCount})-----------")

        dl_train = DataLoader(dataset=ds_train, batch_size=args.mbsz, shuffle=True,
                              num_workers=args.nwork, prefetch_factor=args.mbsz,
                              drop_last=True, pin_memory=True)

        dl_valid = DataLoader(dataset=ds_valid, batch_size=args.mbsz, shuffle=False,
                              num_workers=args.nwork, prefetch_factor=args.mbsz,
                              drop_last=False, pin_memory=True)

        logging.info(f"Loaded {len(ds_train)} training / {len(ds_valid)} validation samples.")

        X_channels = ds_train.X.shape[1]
        model = CNNSR(l_ch_nrs, l_ker_size, l_act_func, X_channels)
        model = nn.DataParallel(model)

        if args.loadChkpt not in [None, "None", "none"]:
            model.load_state_dict(torch.load(args.loadChkpt, map_location=torch_devs))
            model = model.to(torch_devs)
            logging.info(f"Loaded pre-trained model: {args.loadChkpt}")
        else:
            model = model.to(torch_devs)

            def init_weights(m):
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
            model.apply(init_weights)
            logging.info("Initialized model weights using Kaiming normal distribution")

        if args.lossF == "mse": criterion = torch.nn.MSELoss()
        if args.lossF == "mae": criterion = torch.nn.L1Loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        time_on_training = 0

        for epoch in range(args.ecItr):
            ep_tick = time.time()
            time_comp = 0

            pred_train, gt_train, train_loss = [], [], []
            for X_mb, y_mb in dl_train:
                it_comp_tick = time.time()

                optimizer.zero_grad()
                _pred = model.forward(X_mb.to(torch_devs))
                loss = criterion(_pred, y_mb.to(torch_devs))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())

                time_comp += 1000 * (time.time() - it_comp_tick)
                pred_train.append(_pred.cpu().detach().numpy())
                gt_train.append(y_mb.numpy())

            time_e2e = 1000 * (time.time() - ep_tick)
            time_on_training += time_e2e

            logging.info("---------------------------------")
            logging.info(f"Epoch: {epoch}, loss: {np.mean(train_loss)}, elapse: {round(time_e2e,3)}ms/epoch (computation={round(time_comp,3)}ms/epoch, {round(100*time_comp/time_e2e,2)}%)")

            pred_val, gt_val = [], []
            for X_mb_val, y_mb_val in dl_valid:
                with torch.no_grad():
                    _pred = model.forward(X_mb_val.to(torch_devs))
                    pred_val.append(_pred.cpu().numpy())
                    gt_val.append(y_mb_val.numpy())

            pred_val = np.concatenate(pred_val, axis=0)
            gt_val = np.concatenate(gt_val, axis=0)
            pred_train = np.concatenate(pred_train, axis=0)
            gt_train = np.concatenate(gt_train, axis=0)

            l2norm_train = np.sqrt((gt_train[:, 0] - pred_train[:, 0])**2)
            l2norm_val = np.sqrt((gt_val[:, 0] - pred_val[:, 0])**2)

            logging.info('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f ' % (
                (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5)))))

            logging.info('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f ' % (
                (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5)))))

            model_savepath = f"{mod_dir}mod-it{epoch}.pth"
            torch.save(model.state_dict(), model_savepath)

        if loss.cpu().detach().numpy() < args.ecVal:
            b_trainContinue = True

    logging.info("---------------------------------\n")
    logging.info("--------- MODEL CONVERGENCE VERIFIED -----------\n")
    logging.info("---------------------------------")

    for epoch in range(args.ecItr, args.maxItr):
        ep_tick = time.time()
        time_comp = 0

        pred_train, gt_train, train_loss = [], [], []
        for X_mb, y_mb in dl_train:
            it_comp_tick = time.time()

            optimizer.zero_grad()
            _pred = model.forward(X_mb.to(torch_devs))
            loss = criterion(_pred, y_mb.to(torch_devs))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())

            time_comp += 1000 * (time.time() - it_comp_tick)
            pred_train.append(_pred.cpu().detach().numpy())
            gt_train.append(y_mb.numpy())

        time_e2e = 1000 * (time.time() - ep_tick)
        time_on_training += time_e2e

        logging.info("---------------------------------")
        logging.info(f"Epoch: {epoch}, loss: {np.mean(train_loss)}, elapse: {round(time_e2e,3)}ms/epoch (computation={round(time_comp,3)}ms/epoch, {round(100*time_comp/time_e2e,2)}%)")

        pred_val, gt_val = [], []
        for X_mb_val, y_mb_val in dl_valid:
            with torch.no_grad():
                _pred = model.forward(X_mb_val.to(torch_devs))
                pred_val.append(_pred.cpu().numpy())
                gt_val.append(y_mb_val.numpy())

        pred_val = np.concatenate(pred_val, axis=0)
        gt_val = np.concatenate(gt_val, axis=0)
        pred_train = np.concatenate(pred_train, axis=0)
        gt_train = np.concatenate(gt_train, axis=0)

        l2norm_train = np.sqrt((gt_train[:, 0] - pred_train[:, 0])**2)
        l2norm_val = np.sqrt((gt_val[:, 0] - pred_val[:, 0])**2)

        logging.info('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f' % (
            (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5)))))

        logging.info('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f' % (
            (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5)))))

        model_savepath = f"{mod_dir}mod-it{epoch}.pth"
        torch.save(model.state_dict(), model_savepath)

    logging.info("---------------------------------\n")
    logging.info("\n----------- Training finished -----------")
    logging.info("Trained for %3d epochs, each with %d steps (mini-batch size=%d). Total time: %.3f seconds\n" % (
        args.maxItr, len(dl_train), args.mbsz, time_on_training * 1e-3))
