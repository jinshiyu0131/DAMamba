import numpy as np
import seaborn as sns
import cv2
import scipy.io as sio
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

# method = ['BNM', 'DAN', 'DAAN', 'DANN', 'DeepCoral', 'DSAN', 'TSTNET', 'Ours','gt']
# method = ['FDA','MSC','FDA+MSC','gt']
# method = [ 'TSTNET', 'Ours']
# method = [ 'JAN', 'MCC', 'MDD', 'Ours']
# method = ['gt']
# method = ['DAN','DANN', 'DeepCoral']
method3 = ['src_Hangzhou_[92.86875]', 'src_Hangzhou_[92.97961957]', 'src_Hangzhou_[93.31766304]', 'src_Hangzhou_[93.52445652]', 'src_Hangzhou_[94.22309783]']
method2 = ['src_Dioni_[67.53901328]', 'src_Dioni_[67.94610836]', 'src_Dioni_[68.00426481]', 'src_Dioni_[69.49694679]', 'src_Dioni_[69.73926529]']
method1 = ['src_Houston13_[78.30263158]', 'src_Houston13_[78.53947368]', 'src_Houston13_[79.05075188]', 'src_Houston13_[79.17481203]', 'src_Houston13_[79.40977444]']

method = ['src_NC16_[90.78411509]_0.81111', 'src_NC16_[91.17097149]_0.81111',
          'src_NC16_[92.16632786]_0.81111', 'src_NC16_[94.22978862]_0.81111',
          'src_NC16_[94.27874058]_0.81111']
palette = None
# dataset = ['Hangzhou', 'Houston', 'HyRANK']
# dataset = ['HyRANK']
dataset = ['NC16']
# dataset = ['Houston']
# dataset = ['Hangzhou']
#
# out_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/Results/new_re/'
# input_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/Results/new_re/'

# out_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/new_re/Re_maps/'
# input_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/new_re/Re_maps/'

# out_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/new_re/re/Ablation/main_re/Map/'
# input_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/new_re/re/Ablation/main_re/Map/'

# out_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/Results/PJANet_r1/NC16_cls_rgb_map/'
# input_path = '/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/Results/PJANet_r1/NC16_cls_rgb_map/'

out_path = '/mnt/home/duanph_data/transferlearning/code/DeepDA/Results/jsy_visual/mamba/YRD'
input_path = '/mnt/home/duanph_data/transferlearning/code/DeepDA/Results/jsy_visual/mamba/YRD'


for i in range(len(dataset)):
    palette = None
    # img_dir = input_path + dataset[i] + '/' + 'gt' + '.mat'
    # gtt = sio.loadmat(img_dir)['map']
    for j in range(len(method)):
        if dataset[i]=='Houston':
            n_class = 8
            palette = {0: (0,0,0),1: (0, 31, 255), 2: (0, 175, 255), 3: (63, 255, 191),
                     4: (219, 255, 41), 5: (255, 159, 0), 6: (255, 15, 0),7: (127, 0, 0)}
            img_dir = input_path + '/' + method[j] + '.npy'
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

            out_name = out_path + '/' + method[j] + '.png'
            # out_name = out_path + dataset[i] + '/' + method[j] + '.png'
            cv2.imwrite(out_name, color_img)
        elif dataset[i]=='HyRANK':
            n_class = 13
            palette = {0: (0,0,0),1:(0,0,223), 2:(0,54,255), 3:(0,146,255), 4:(0,223,255), 5:(47,255,207), 6:(143,255,111),
                       7:(223,255,31), 8:(255,207,0), 9:(255,113,0), 10:(255,31,0), 11:(207,0,0), 12:(127,0,0), }
            img_dir = input_path + '/' + method[j] + '.npy'
            img = np.load(img_dir) + 1
            img = img[7:-7, 7:-7]
            # for k in range(n_class):
            #     patch = np.zeros([50, 50])
            #     patch = patch + k
            #     color_img = convert_to_color_(patch, palette)
            #     out_name = out_path + dataset[i] + '/' + str(k)+ '.png'
            #     cv2.imwrite(out_name, color_img)
            color_img = convert_to_color_(img, palette)

            out_name = out_path + '/' + method[j] + '.png'
            cv2.imwrite(out_name, color_img)
        elif dataset[i]=='Hangzhou':
            n_class = 4
            palette = {0: (0,0,0),1:(0,0,143), 2:(143,255,111), 3:(127,0,0)}
            img_dir = input_path + '/' + method[j] + '.npy'
            img = np.load(img_dir) + 1
            img = img[7:-7, 7:-7]
            # for k in range(n_class):
            #     patch = np.zeros([50, 50])
            #     patch = patch + k
            #     color_img = convert_to_color_(patch, palette)
            #     out_name = out_path + dataset[i] + '/' + str(k)+ '.png'
            #     cv2.imwrite(out_name, color_img)
            color_img = convert_to_color_(img, palette)

            out_name = out_path + '/' + method[j] + '.png'
            cv2.imwrite(out_name, color_img)
        elif dataset[i]=='NC16':
            n_class = 8
            palette = {0: (0,0,0),1: (0, 0, 139), 2: (128, 128, 240), 3: (139, 0, 0),
                     4: (152, 251, 152), 5: (34, 139, 34), 6: (235, 206, 135),7: (133, 21, 199)}  #34, 139, 34
            img_dir = input_path + '/' + method[j] + '.npy'
            img = np.load(img_dir) + 1
            img = img[7:-7, 7:-7]
            print("Unique values in img:", np.unique(img))
            mask = cv2.imread('/mnt/home/duanph_data/transferlearning/code/DeepDA/mask.png', cv2.IMREAD_GRAYSCALE)
            print("Unique values in mask:", np.unique(mask))
            np.save('/mnt/home/duanph_data/transferlearning/code/DeepDA//mask.npy', mask)
            mask = np.load('/mnt/home/duanph_data/transferlearning/code/DeepDA//mask.npy')
            img[mask==255] = 0
            # invalid_mask = img==0
            # img[invalid_mask] = 0
            # mask = np.load('/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/Results/PJANet_r1/NC16_cls_rgb_map/NC16/mask.npy')
            # img[mask == 255] = 0
            color_img = convert_to_color_(img, palette)
            out_name = out_path + '/' + method[j] + '.png'
            cv2.imwrite(out_name, color_img)
        #     mask = np.load('/home/xzj/program/Code/HSI_DA/transferlearning/code/DeepDA/Results/PJANet_r1/NC16_cls_rgb_map/NC16/mask.npy')
        # if method[j] =='gt':
        #     img_dir = input_path + dataset[i] + '/' +  method[j] + '.mat'
        #     img = sio.loadmat(img_dir)['map']
        #     # img[img !=3] = 0
        # else:
        #     img_dir = input_path + dataset[i] + '/' + method[j] + '.npy'
        #     img = np.load(img_dir) + 1
        # if method[j] is not 'gt':
        #     img = img[7:-7, 7:-7]
        #     if dataset[i] == 'NC16':
        #         img[mask==255]=0
                # img[mask!=1]=0
        # if dataset[i]=='Hangzhou':
        #     img = img
        # else:
        #     img[gtt==0]=0

        # color_img = convert_to_color_(img, palette)
        #
        # out_name = out_path + dataset[i] + '/' + method[j] + '.png'
        # cv2.imwrite(out_name, color_img)





# import scipy.io as sio
# import cv2
# ###############高光谱数据伪彩色图像################
# hu18 = sio.loadmat('/home/xzj/program/DataSets/域自适应/Houston/Houston18.mat')['ori_data'][:,:,(23,11,6)]
# hu13 = sio.loadmat('/home/xzj/program/DataSets/域自适应/Houston/Houston13.mat')['ori_data'][:,:,(23,11,6)]
# # sh = sio.loadmat('/home/xzj/program/DataSets/域自适应/Shanghai-Hangzhou/Shanghai.mat')['ori_data'][:, :, (29,23,16)]
# # hz = sio.loadmat('/home/xzj/program/DataSets/域自适应/Shanghai-Hangzhou/Hangzhou.mat')['ori_data'][:,:,(50,23,6)]
# # dioni = sio.loadmat('/home/xzj/program/DataSets/域自适应/HyRANK/HyRANK-matlab/Dioni.mat')['ori_data'][:,:,(50,23,6)]
# # lokia = sio.loadmat('/home/xzj/program/DataSets/域自适应/HyRANK/HyRANK-matlab/Loukia.mat')['ori_data'][:,:,(50,23,6)]
#
# hu18 = hu18 / hu18.max()
# hu13 = hu13 / hu13.max()
# # sh = sh / sh.max()
# # hz = hz / hz.max()
# # dioni = dioni / dioni.max()
# # lokia = lokia / lokia.max()
#
# cv2.imwrite('hu18.png', hu18*255*2)
# cv2.imwrite('hu13.png', hu13*255*2)
# # cv2.imwrite('sh.png', sh*255)
# # cv2.imwrite('hz.png', hz*255)
# # cv2.imwrite('dioni.png', dioni*255)
# # cv2.imwrite('lokia.png', lokia*255)