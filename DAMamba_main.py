import time

import configargparse
import torch
import DAMamba_model
import utils
from utils import str2bool
import numpy as np
import random
from hsi_dataset import get_dataset, HyperX, HyperX_w, data_prefetcher
from hsi_uti import sample_gt, metrics, get_device, seed_worker, count_sliding_window, sliding_window, grouper
import torch.utils.data as data
from losses import loss_unl, Prototype_t

from tqdm import tqdm
import numpy as np
import seaborn as sns
import cv2
import scipy.io as sio
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--start_iter', type=int, default=0, required=True)
    parser.add_argument('--save_path', type=str, default=None, required=True)
    parser.add_argument('--start_msc', type=int, default=20)
    # training related
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")
    parser.add_argument('--noise_std', type=float, default=0.1)

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')

    # proml related
    parser.add_argument('--warm_steps', type=int, default=10)
    parser.add_argument('--T', type=float, default=0.1, metavar='T',
                        help='temperature (default: 0.1)')
    parser.add_argument('--threshold1', default=0.95, type=float,
                        help='pseudo label threshold1')
    parser.add_argument('--threshold2', default=0.4, type=float,
                        help='pseudo label threshold2')
    parser.add_argument('--cls_weight', default=0.8, type=float)
    parser.add_argument('--fixmatch_weight', default=1, type=float)
    parser.add_argument('--inter_weight', default=1, type=float)
    parser.add_argument('--intra_weight', default=1, type=float)
    parser.add_argument('--batch_weight', default=1, type=float)
    parser.add_argument('--re_ratio', default=5, type=int)
    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    training_sample_ratio = 0.05
    re_ratio = args.re_ratio  # sh:2 10
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.src_domain,
                                                                                        args.data_dir)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.tgt_domain,
                                                                                        args.data_dir)

    n_class = gt_src.max()
    N_BANDS = img_src.shape[-1]
    DEVICE = get_device(0)
    hyperparams = vars()
    hyperparams.update({'n_classes': n_class, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full',
                        'flip_augmentation':True, 'radiation_augmentation':True, 'mixture_augmentation':True,
                        'patch_size':12, 'batch_size':100})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = training_sample_ratio * re_ratio * sample_num_src / sample_num_tar
    training_sample_tar_ratio = tmp if tmp < 1 else 1

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, _, training_set, _ = sample_gt(gt_src, training_sample_ratio, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    train_gt_tar, _, _, _ = sample_gt(gt_tar, training_sample_tar_ratio, mode='random')
    img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar

    if tmp < 1:
        for i in range(re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            # img_tar_con = np.concatenate((img_tar_con,img_tar))
            # train_gt_tar_con = np.concatenate((train_gt_tar_con,train_gt_tar))

    hyperparams_train = hyperparams.copy()

    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)

    g = torch.Generator()
    g.manual_seed(args.seed)
    source_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True,
                                   drop_last=True,
                                    num_workers=1)
    train_tar_s_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
    train_tar_w_dataset = HyperX_w(img_tar_con, train_gt_tar_con, **hyperparams_train)
    target_train_s_loader = data.DataLoader(train_tar_s_dataset,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       batch_size=hyperparams['batch_size'],
                                       shuffle=False,
                                       drop_last=True,
                                          num_workers=1)
    target_train_w_loader = data.DataLoader(train_tar_w_dataset,
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=1)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    target_test_loader = data.DataLoader(test_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=False,
                                        drop_last=True,
                                        batch_size=hyperparams['batch_size'],
                                         num_workers=1)

    return source_loader, target_train_s_loader, target_train_w_loader, target_test_loader, n_class, N_BANDS, img_tar, gt_tar, LABEL_VALUES_tar



def get_model(n_bands,args):
    model = jsy_model.TransferNet(
        n_bands,args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck, proto_t=Prototype_t(args.n_class)).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            time1 = time.time()
            data, target = data.to(args.device), (target.to(args.device).type(torch.int64)) - 1
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            time2 = time.time()
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_s_loader, target_train_w_loader,target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_s_loader = len(target_train_s_loader)
    len_target_w_loader = len(target_train_w_loader)
    n_batch = min(len_source_loader, len_target_s_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    # iter_source, iter_target_s, iter_target_w = iter(source_loader), iter(target_train_s_loader), iter(target_train_w_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch + 1):
        time1 = (time.time())
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        #
        if max(len_target_s_loader, len_target_w_loader, len_source_loader) != 0:
            iter_source, iter_target_s, iter_target_w = iter(source_loader), iter(target_train_s_loader), iter(target_train_w_loader)

        # criterion = torch.nn.CrossEntropyLoss()
        proto_t = Prototype_t(C=args.n_class, dim=256)
        for bbb in range(n_batch):

            data_source, label_source = next(iter_source)  # .next()
            data_target_s, _ = next(iter_target_s)  # .next()
            data_target_w, _ = next(iter_target_w)
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target_s = data_target_s.to(args.device)
            data_target_w = data_target_w.to(args.device)


            label_source = label_source.type(torch.int64) - 1
            L_base, L_batch, L_fixmatch, L_inter, L_intra, unl_mask, unl_pseudo_label, feat_tu_w, feat_tu_s = model(bbb,
                                                                                                                    proto_t,
                                                                                                                    data_source,
                                                                                                                    data_target_s,
                                                                                                                    data_target_w,
                                                                                                                    label_source,
                                                                                                                    e,
                                                                                                                    args.start_msc,
                                                                                                                    args)
            # clf_loss, transfer_loss, msc_loss = model(data_source, data_target_s, data_target_w, label_source, e, args.start_msc)
            # loss = clf_loss + args.transfer_loss_weight * transfer_loss
            proml_loss = L_fixmatch * args.fixmatch_weight + L_inter * args.inter_weight + L_intra * args.intra_weight + L_batch * args.batch_weight
            loss = L_base * args.cls_weight
            if e > args.start_msc:
                if not torch.isnan(proml_loss):
                    loss = loss + proml_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            proto_t.update(feat_tu_w, bbb, unl_mask, unl_pseudo_label, args, norm=True)
            train_loss_clf.update(L_base.item())
            # train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        time2 = (time.time())
        print(time2-time1)
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
            model_name = args.save_path + args.src_domain + f"_{args.lr}_{args.cls_weight}_{args.fixmatch_weight}_{args.inter_weight}_{args.intra_weight}_{args.batch_weight}" + '_best_model.pth'
            torch.save(model, model_name)
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: {:.4f}'.format(best_acc))

def test_model(net, img, patch_size, n_classes):
    """
    Test a model on a specific image
    """
    net.eval()
    center_pixel = True
    batch_size, device = 100, 'cuda'


    kwargs = {
        "step": 1,
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)


            indices = [b[1:] for b in batch]
            data = data.to(device)

            output = net.predict(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs
def test_hsi(img, gt_tgt, label_values, model, patch, classes):
    # img = np.pad(img, [[6], [6], [0]], mode='symmetric')
    probabilities = test_model(model, img, patch_size=patch, n_classes=classes)
    # probabilities = probabilities[6:-6, 6:-6, :]
    prediction = np.argmax(probabilities, axis=-1)
    gt_tgt = gt_tgt.astype(np.int64) - 1
    run_results = metrics(
        prediction,
        gt_tgt,
        ignored_labels=[-1],
        n_classes=len(label_values),
    )
    print(run_results["Accuracy"])

    return run_results, prediction

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d



def main():
    parser = get_parser()
    args = parser.parse_args()
    iter_num = 10
    if args.src_domain == 'Houston13':
        A = np.zeros([iter_num, 7])
        img_save_path = args.save_path
        # seed = [16, 19, 5, 9, 0]
        seed = [0, 1, 5, 9, 10, 11, 12, 16, 17, 19]  # houston
        # seed = [2, 3, 4, 6, 7, 8, 13, 14, 15, 18]
        # seed = [10,]  ##houston
    elif args.src_domain == 'Dioni':
        A = np.zeros([iter_num, 12])
        img_save_path = args.save_path
        seed = [0,1,2,3,4,5,6,7,8,9]
    elif args.src_domain == 'Hangzhou':
        A = np.zeros([iter_num, 3])
        img_save_path = args.save_path
        seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif args.src_domain == 'NC16':
        A = np.zeros([iter_num, 7])
        img_save_path = args.save_path
        seed = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    acc = np.zeros([iter_num, 1])  # 10
    k = np.zeros([iter_num, 1])
    # seed = [0,1,5,9,10,11,12,16,17,19]##houston
    method = []
    for i in range(args.start_iter, iter_num):  # 10
        j = seed[i]   # i
        best_acc = 0
        setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(args)
        set_random_seed(j)
        source_loader, target_train_s_loader, target_train_w_loader, target_test_loader, n_class, n_bands, img_tar, gt_tar, LABEL_VALUES = load_data(args)
        setattr(args, "n_class", n_class)
        if args.epoch_based_training:
            setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_s_loader)))
        else:
            setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
        model = get_model(n_bands,args)
        optimizer = get_optimizer(model, args)

        if args.lr_scheduler:
            scheduler = get_scheduler(optimizer, args)
        else:
            scheduler = None
        # args.n_epoch = 500

        train(source_loader, target_train_s_loader, target_train_s_loader,target_test_loader, model, optimizer, scheduler, args)
        model_na = args.save_path + args.src_domain + f"_{args.lr}_{args.cls_weight}_{args.fixmatch_weight}_{args.inter_weight}_{args.intra_weight}_{args.batch_weight}" + '_best_model.pth'
        model = torch.load(model_na)
        results, pre = test_hsi(img_tar, gt_tar, LABEL_VALUES, model, 12, n_class)
        acc[i] = results['Accuracy']
        A[i, :] = results['F1 scores']
        k[i] = results['Kappa']
        input_path = '/mnt/home/duanph_data/transferlearning/code/DeepDA/Results/jsy_visual/mamba/'
        img_path = input_path + 'src_' + args.src_domain + '_' + str(acc[i]) + f"_{args.cls_weight}{args.fixmatch_weight}{args.inter_weight}{args.intra_weight}{args.batch_weight}" + '.npy'
        method.append(img_path)
        print(results)
        if results['Accuracy'] > best_acc:
            best_acc = results['Accuracy']
            np.save(img_path, pre)
    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)
    print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
    print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
    print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
    print("accuracy for each class: ")
    for i in range(n_class):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

    f_txt = img_save_path + args.src_domain + '_' + 'results.txt'
    fw = open(f_txt, 'w')
    fw.write('OA:' + str(round(OAMean, 4)) + '\n' + 'AA:' + str(round(AAMean, 4))
             + '\n' + 'Kappa:' + str(round(kMean, 4))
             + '\n' + '+-oa:' + str(np.round((OAStd), 4))
             + '\n' + '+-aa:' + str(np.round((AAStd), 4))
             + '\n' + '+-kappa:' + str(np.round((kStd), 4))
             )
    fw.close()


    palette = None
    dataset = ['Houston']
    out_path = '/mnt/home/duanph_data/transferlearning/code/DeepDA/Results/jsy_visual/mamba/'
    input_path = '/mnt/home/duanph_data/transferlearning/code/DeepDA/Results/jsy_visual/mamba/'
    for i in range(len(dataset)):
        palette = None
        # img_dir = input_path + dataset[i] + '/' + 'gt' + '.mat'
        # gtt = sio.loadmat(img_dir)['map']
        for j in range(len(method)):
            if dataset[i] == 'Houston':
                n_class = 8
                palette = {0: (0, 0, 0), 1: (0, 31, 255), 2: (0, 175, 255), 3: (63, 255, 191),
                           4: (219, 255, 41), 5: (255, 159, 0), 6: (255, 15, 0), 7: (127, 0, 0)}
                img_dir = method[j]
                # img_dir = input_path + '/' + method[j] + '.npy'
                # img_dir = input_path + dataset[i] + '/' + method[j] + '.npy'
                img = np.load(img_dir) + 1
                img = img[7:-7, 7:-7]
                # for k in range(n_class):
                #     patch = np.zeros([50, 50])
                #     patch = patch + k
                #     color_img = convert_to_color_(patch, palette)
                #     out_name = out_path + dataset[i] + '/' + str(k)+ '.png'
                #     cv2.imwrite(out_name, color_img)
                color_img = convert_to_color_(img, palette)

                out_name = method[j] + '.png'
                # out_name = out_path + dataset[i] + '/' + method[j] + '.png'
                cv2.imwrite(out_name, color_img)


if __name__ == "__main__":
    main()

