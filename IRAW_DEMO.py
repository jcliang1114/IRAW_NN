from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from scipy.optimize import basinhopping
import os
import cv2
import random

__version__ = '0.0.12'


im_watermark = Image.open('.\\AIR_32.png')


watermark_path = '.\\IRAW_Results'
true_list = [1, 9, 14, 18, 23, 30, 39, 42, 51, 52, 53, 59, 72, 74, 93, 106, 129, 132, 138, 142, 144, 152, 157, 159, 167,
             172, 176, 194, 195]
classes = ['65', '970', '230', '809', '516', '57', '334', '415', '674', '332', '109', '286', '370', '757', '595', '147',
           '108', '23', '478', '517', '334', '173', '948', '727', '23', '846', '270', '167', '55', '858', '324', '573',
           '150', '981', '586', '887', '32', '398', '777', '74', '516', '756', '129', '198', '256', '725', '565', '167',
           '717', '394', '92', '29', '844', '591', '358', '468', '259', '994', '872', '588', '474', '183', '107', '46',
           '842', '390', '101', '887', '870', '841', '467', '149', '21', '476', '80', '424', '159', '275', '175', '461',
           '970', '160', '788', '58', '479', '498', '369', '28', '487', '50', '270', '383', '366', '780', '373', '705',
           '330', '142', '949', '349', '473', '159', '872', '878', '201', '906', '70', '486', '632', '608', '122',
           '720', '227', '686', '173', '959', '638', '646', '664', '645', '718', '483', '852', '392', '311', '457',
           '352', '22', '934', '283', '802', '553', '276', '236', '751', '343', '528', '328', '969', '558', '163',
           '328', '771', '726', '977', '875', '265', '686', '590', '975', '620', '637', '39', '115', '937', '272',
           '277', '763', '789', '646', '213', '493', '647', '504', '937', '687', '781', '666', '583', '158', '825',
           '212', '659', '257', '436', '196', '140', '248', '339', '230', '361', '544', '935', '638', '627', '289',
           '867', '272', '103', '584', '180', '703', '449', '771', '118', '396', '934', '16', '548', '993', '704',
           '457', '233', '401', '827', '376', '146', '606', '922', '516', '284', '889', '475', '978', '475', '984',
           '16', '77', '610', '254', '636', '662', '473', '213', '25', '463', '215', '173', '35', '741', '125', '787',
           '289', '425', '973', '1', '167', '121', '445', '702', '532', '366', '678', '764', '125', '349', '13', '179',
           '522', '493', '989', '720', '438', '660', '983', '533', '487', '27', '644', '750', '865', '1', '176', '694',
           '695', '798', '925', '413', '250', '970', '821', '421', '361', '920', '761', '27', '676', '92', '194', '897',
           '612', '610', '283', '881', '906', '899', '426', '554', '403', '826', '869', '730', '0', '866', '580', '888',
           '43', '64', '69', '176', '329', '469', '292', '991', '591', '346', '1', '607', '934', '784', '572', '389',
           '979', '654', '420', '390', '702', '24', '102', '949', '508', '361', '280', '65', '777', '359', '165', '21',
           '7', '525', '760', '938', '254', '733', '707', '463', '60', '887', '531', '380', '982', '305', '355', '503',
           '818', '495', '472', '293', '816', '195', '904', '475', '481', '431', '260', '130', '627', '460', '622',
           '696', '300', '37', '133', '637', '675', '465', '592', '741', '895', '91', '109', '582', '694', '546', '208',
           '488', '786', '959', '192', '834', '879', '649', '228', '621', '630', '107', '598', '420', '134', '133',
           '185', '471', '230', '974', '74', '76', '852', '383', '267', '419', '359', '484', '510', '33', '177', '935',
           '310', '998', '270', '598', '199', '998', '836', '14', '97', '856', '398', '319', '549', '92', '765', '412',
           '945', '160', '265', '638', '619', '722', '183', '674', '468', '468', '885', '675', '636', '196', '912',
           '721', '16', '199', '175', '775', '944', '350', '557', '361', '361', '594', '861', '257', '606', '734',
           '767', '746', '788', '346', '153', '739', '414', '915', '940', '152', '943', '849', '712', '100', '546',
           '744', '764', '141', '39', '993', '758', '190', '888', '18', '584', '341', '875', '359', '388', '894', '437',
           '987', '517', '372', '286', '662', '713', '915', '964', '146', '529', '416', '376', '147', '902', '26',
           '398', '175', '270', '335', '532', '540', '607', '495', '222', '801', '982', '304', '166', '56', '868',
           '448', '744', '567', '277', '298', '651', '377', '684', '832', '39', '219', '863', '868', '794', '80', '983',
           '269', '238', '498', '223', '521', '830', '260', '491', '896', '220', '680', '48', '542', '961', '820',
           '148', '114', '99', '143', '691', '796', '986', '346', '367', '939', '875', '625', '482', '464', '812',
           '705', '860', '466', '781', '499', '338', '488', '858', '795', '437', '11', '625', '965', '874', '928',
           '600', '86', '133', '149', '865', '480', '325', '499', '834', '421', '298', '900', '905', '184', '740',
           '258', '762', '295', '129', '240', '833', '471', '385', '899', '162', '288', '450', '850', '227', '273',
           '954', '965', '611', '643', '147', '290', '866', '186', '156', '776', '775', '998', '333', '325', '572',
           '927', '744', '777', '833', '551', '301', '716', '848', '102', '790', '959', '404', '987', '415', '455',
           '242', '600', '517', '16', '320', '632', '568', '338', '216', '331', '726', '959', '605', '134', '677',
           '288', '10', '718', '852', '440', '104', '712', '388', '261', '609', '620', '341', '579', '450', '628',
           '217', '878', '763', '209', '126', '663', '864', '232', '776', '942', '336', '733', '681', '512', '78',
           '668', '699', '746', '46', '618', '330', '615', '427', '62', '116', '127', '955', '306', '425', '190', '370',
           '187', '971', '534', '397', '657', '840', '718', '116', '836', '994', '419', '764', '214', '285', '641',
           '951', '882', '13', '829', '624', '216', '665', '521', '268', '468', '418', '728', '356', '449', '194',
           '362', '948', '924', '249', '524', '992', '571', '283', '608', '129', '486', '859', '498', '21', '467',
           '591', '924', '556', '97', '898', '586', '10', '202', '67', '501', '141', '603', '727', '101', '995', '278',
           '964', '240', '423', '634', '533', '424', '451', '555', '732', '514', '803', '300', '551', '753', '411',
           '315', '963', '129', '389', '601', '526', '839', '299', '578', '112', '960', '632', '867', '273', '61',
           '427', '367', '924', '413', '34', '773', '654', '131', '874', '282', '891', '956', '201', '267', '969',
           '200', '673', '423', '907', '57', '27', '459', '863', '322', '934', '663', '424', '687', '837', '958', '645',
           '120', '306', '930', '121', '694', '524', '205', '137', '849', '681', '380', '586', '916', '478', '182',
           '874', '715', '590', '111', '19', '161', '915', '730', '678', '822', '818', '699', '601', '673', '233',
           '501', '624', '679', '400', '581', '665', '903', '622', '585', '800', '899', '669', '81', '746', '595',
           '935', '668', '295', '893', '266', '628', '987', '367', '294', '727', '12', '876', '186', '589', '70', '129',
           '454', '17', '946', '200', '181', '163', '80', '940', '587', '21', '198', '25', '932', '339', '480', '764',
           '883', '454', '807', '287', '868', '614', '814', '591', '919', '508', '479', '452', '155', '41', '163',
           '606', '7', '269', '576', '858', '506', '23', '447', '397', '595', '753', '5', '186', '667', '305', '46',
           '303', '673', '927', '91', '34', '757', '406', '390', '76', '517', '806', '330']

scale_size = 0.8


NP = 100
CR = 0.9
GENERATIONS = 6
N_ITER = 3
MUTATION_COEFF_INIT = 0.5
MUTATION_COEFF_FINAL = 0.0


def plus(str):
    return str.zfill(8)


# 密钥生成函数
def generate_keys(xs):
    S = max(0, min(99, int(xs[3])))
    K = max(0, min(99, int(xs[4])))
    L = max(2, min(5, int(xs[5])))
    P = max(2, min(5, int(xs[6])))

    T1 = (S * 13 + K * 7) % 256
    T2 = (K * 17 + S * 3) % 256
    T3 = (L * 19 + P * 11) % 32
    T4 = (P * 23 + L * 5) % 32

    Key1 = (T1 * 31 + S) % 1024
    Key2 = (T2 * 43 + K) % 1024
    Key3 = (T3 * 59 + L) % 64
    Key4 = (T4 * 61 + P) % 64

    u = (Key3 % 4) + 2
    v = (Key4 % 4) + 2

    return Key1, Key2, Key3, Key4, u, v


# 宿主图像加密
def encrypt_host_image(img, key):
    np.random.seed(key)
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    indices = np.random.permutation(h * w).reshape(h, w)
    for c in range(img_np.shape[2]):
        flat = img_np[:, :, c].flatten()
        img_np[:, :, c] = flat[indices.flatten()].reshape(h, w)
    return Image.fromarray(img_np)


# 水印加密
def encrypt_watermark(wm, key):
    np.random.seed(key)
    wm_np = np.array(wm)
    h, w = wm_np.shape[:2]
    a = np.random.randint(1, 4)
    b = np.random.randint(1, 4)
    iterations = np.random.randint(3, 7)

    for _ in range(iterations):
        new_wm = np.zeros_like(wm_np)
        for x in range(h):
            for y in range(w):
                new_x = (x + a * y) % h
                new_y = (b * x + (a * b + 1) * y) % w
                new_wm[new_x, new_y] = wm_np[x, y]
        wm_np = new_wm
    return Image.fromarray(wm_np)


# 添加水印
def add_watermark_to_image(image, xs, watermark, sl, xuhao):
    s = max(0, min(15, int(xs[0])))

    Key1, Key2, Key3, Key4, u_encrypted, v_encrypted = generate_keys(xs)

    encrypted_host = encrypt_host_image(image, Key1)

    original_wm_width, original_wm_height = watermark.size

    ref_threshold = (original_wm_width * original_wm_height) // 2
    current_w, current_h = original_wm_width, original_wm_height

    if current_w * current_h > ref_threshold:
        adjusted_w = original_wm_width // 2
        adjusted_h = original_wm_height // 2
        watermark = watermark.resize((adjusted_w, adjusted_h), resample=Image.ANTIALIAS)
    else:
        adjusted_w, adjusted_h = current_w, current_h

    encrypted_wm = encrypt_watermark(watermark, Key2)

    rgba_image = encrypted_host.convert('RGBA')
    rgba_watermark = encrypted_wm.convert('RGBA')

    host_scale = 1.0 + s * 0.05
    orig_w, orig_h = rgba_image.size
    new_host_size = (int(orig_w * host_scale), int(orig_h * host_scale))
    rgba_image = rgba_image.resize(new_host_size, resample=Image.ANTIALIAS)

    wm_scale = sl * (1.0 + s * 0.03)
    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size
    watermark_scale = min(image_x / (wm_scale * watermark_x), image_y / (wm_scale * watermark_y))
    new_wm_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    rgba_watermark = rgba_watermark.resize(new_wm_size, resample=Image.ANTIALIAS)

    a = np.array((int(xs[1]) + 112) % 112)
    b = np.array((int(xs[2]) + 112) % 112)
    k = int(xs[9] * 224 / 50 + 0.5)
    if k < 78:
        k = 78
    x_pos = int(a)
    y_pos = int(b)

    class WaterMark:
        def __init__(self, password_wm=x_pos, password_img=y_pos, block_shape=(8, 8)):
            self.block_shape = np.array(block_shape)
            self.password_wm = password_wm
            self.password_img = password_img
            self.u = u_encrypted
            self.v = v_encrypted
            self.d1, self.d2 = 36, 20

            self.img = None
            self.img_shape = None
            self.block_count_h = 0
            self.block_count_w = 0
            self.block_num = 0
            self.wm_size = 0
            self.wm_bit = None
            self.actual_wm_shape = (adjusted_w, adjusted_h)

        def read_img(self, filename):
            self.img = cv2.imread(filename).astype(np.float32)
            self.img_shape = self.img.shape[:2]

            pad_h = (self.block_shape[0] - self.img_shape[0] % self.block_shape[0]) % self.block_shape[0]
            pad_w = (self.block_shape[1] - self.img_shape[1] % self.block_shape[1]) % self.block_shape[1]
            self.img = cv2.copyMakeBorder(self.img, 0, pad_h, 0, pad_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

            self.block_count_h = self.img.shape[0] // self.block_shape[0]
            self.block_count_w = self.img.shape[1] // self.block_shape[1]
            self.block_num = self.block_count_h * self.block_count_w

        def read_wm(self, filename):
            wm_img = cv2.imread(filename)[:, :, 0]
            self.wm_bit = (wm_img.flatten() > 128).astype(np.int8)
            self.wm_size = self.wm_bit.size
            if self.wm_size > self.block_num:
                self.wm_bit = self.wm_bit[:self.block_num]
                self.wm_size = self.block_num

        def block_add_wm(self, block, wm_bit):
            block_dct = cv2.dct(block.astype(np.float32))

            embed_strength = 5
            if wm_bit == 1:
                block_dct[self.u, self.v] += embed_strength
                if self.u < self.block_shape[0] - 1:
                    block_dct[self.u + 1, self.v] += embed_strength * 0.5
                if self.v < self.block_shape[1] - 1:
                    block_dct[self.u, self.v + 1] += embed_strength * 0.5
            else:
                block_dct[self.u, self.v] -= embed_strength
                if self.u < self.block_shape[0] - 1:
                    block_dct[self.u + 1, self.v] -= embed_strength * 0.5
                if self.v < self.block_shape[1] - 1:
                    block_dct[self.u, self.v + 1] -= embed_strength * 0.5

            return cv2.idct(block_dct)

        def embed(self, output_filename):
            embed_img = self.img.copy()
            wm_idx = 0
            for h in range(self.block_count_h):
                for w in range(self.block_count_w):
                    if wm_idx >= self.block_num:
                        break
                    h_start = h * self.block_shape[0]
                    h_end = h_start + self.block_shape[0]
                    w_start = w * self.block_shape[1]
                    w_end = w_start + self.block_shape[1]
                    block = embed_img[h_start:h_end, w_start:w_end, :]

                    for channel in range(3):
                        block_channel = block[:, :, channel]
                        current_wm_bit = self.wm_bit[wm_idx % self.wm_size]
                        block_embed = self.block_add_wm(block_channel, current_wm_bit)
                        embed_img[h_start:h_end, w_start:w_end, channel] = block_embed

                    wm_idx += 1

            embed_img = embed_img[:self.img_shape[0], :self.img_shape[1], :]
            embed_img = np.clip(embed_img, 0, 255).astype(np.uint8)
            cv2.imwrite(output_filename, embed_img)
            return embed_img

        def extract(self, filename, out_wm_name=None):
            self.wm_size = np.prod(self.actual_wm_shape)
            self.read_img(filename)

            wm_extract = np.zeros(self.wm_size)
            wm_idx = 0
            for h in range(self.block_count_h):
                for w in range(self.block_count_w):
                    if wm_idx >= self.wm_size:
                        break
                    h_start = h * self.block_shape[0]
                    h_end = h_start + self.block_shape[0]
                    w_start = w * self.block_shape[1]
                    w_end = w_start + self.block_shape[1]
                    block = self.img[h_start:h_end, w_start:w_end, :]

                    channel_wm = []
                    for channel in range(3):
                        block_dct = cv2.dct(block[:, :, channel].astype(np.float32))
                        wm_bit = 1 if block_dct[self.u, self.v] > 0 else 0
                        channel_wm.append(wm_bit)
                    wm_extract[wm_idx] = np.mean(channel_wm)
                    wm_idx += 1

            if out_wm_name:
                wm_img = (wm_extract.reshape(self.actual_wm_shape) * 255).astype(np.uint8)
                cv2.imwrite(out_wm_name, wm_img)
            return wm_extract


    wm_temp = Image.open('.\\AIR_32.png').resize((adjusted_w, adjusted_h), 1)
    temp_wm_path = f'.\\pic\\wm_temp_{xuhao}.png'
    wm_temp.save(temp_wm_path)

    bwm1 = WaterMark(password_wm=x_pos, password_img=y_pos)

    img_temp = Image.open(f'.\\Raw Data\\{xuhao}.png').resize((k, k), 1)
    img_temp = img_temp.resize((int(k * host_scale), int(k * host_scale)), 1)
    temp_img_path = f'.\\pic\\img_{k}_s{s}_{xuhao}.png'
    img_temp.save(temp_img_path)
    bwm1.read_img(temp_img_path)
    bwm1.read_wm(temp_wm_path)

    bwm1.embed('.\\output\\dashangshuiyin_raodong.png')
    bwm1.extract('.\\output\\dashangshuiyin_raodong.png',
                 out_wm_name='.\\output\\jiechushuiyin_raodong.png')

    img_final = Image.open('.\\output\\dashangshuiyin_raodong.png').resize((224, 224), 1)
    img_np = np.array(img_final.convert('RGB'))
    h, w = img_np.shape[:2]

    P1 = [s, xs[1], xs[2]]
    P2 = [u_encrypted, v_encrypted, Key1 % 256]
    P3 = [Key2 % 256, Key3 % 256, Key4 % 256]

    P1 = np.clip(P1, 0, 255).astype(np.uint8)
    P2 = np.clip(P2, 0, 255).astype(np.uint8)
    P3 = np.clip(P3, 0, 255).astype(np.uint8)
    if w >= 3:
        img_np[h - 1, w - 3] = P3
        img_np[h - 1, w - 2] = P2
        img_np[h - 1, w - 1] = P1
    img_final = Image.fromarray(img_np)

    img_final.save(f'.\\output\\dashangshuiyin_{k}_s{s}_raodong.png')


    if os.path.exists(temp_wm_path):
        os.remove(temp_wm_path)
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return img_final


backbone_name = 'resnet101'
model = models.__dict__[backbone_name](pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()


def label_model(input):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    input = transform(input)
    input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
    return model(input.cuda())


def predict_classes(img, xs, watermark, target_class, sl, xuhao):
    imgs_perturbed = add_watermark_to_image(img, xs, watermark, sl, xuhao)
    imgs_perturbed = imgs_perturbed.convert('RGB')
    predictions = label_model(imgs_perturbed).cpu().detach().numpy()
    return predictions[0][target_class]


def attack_success(img, xs, watermark, sl, target_class, xuhao, verbose=False):
    attack_image = add_watermark_to_image(img, xs, watermark, sl, xuhao)
    attack_image = attack_image.convert('RGB')
    predict = label_model(attack_image).cpu().detach().numpy()
    predicted_class = np.argmax(predict)
    # ========== 核心修改7：临时攻击图像路径改为相对路径 ==========
    attack_image.save('.\\temp1_VGG.png')

    if verbose:
        print(f'目标类别置信度: {predict[0][target_class]:.4f}')

    return predicted_class != target_class


# BH优化
def bh_optimization(im_before, label, im_watermark, sl, xuhao, x0, mutation_coeff):
    original_wm_width, original_wm_height = im_watermark.size
    ref_threshold = (original_wm_width * original_wm_height) // 2
    current_w, current_h = original_wm_width, original_wm_height

    if current_w * current_h > ref_threshold:
        adjusted_w = original_wm_width // 2
        adjusted_h = original_wm_height // 2
    else:
        adjusted_w, adjusted_h = current_w, current_h

    watermark_scale = min(224 / (sl * adjusted_w), 224 / (sl * adjusted_h))
    watermark_x1 = int(adjusted_w * watermark_scale)
    watermark_y1 = int(adjusted_h * watermark_scale)

    def predict_fn(xs):
        return -predict_classes(im_before, xs, im_watermark, int(label), sl, xuhao)

    def callback_fn(xs, f, accept):
        return attack_success(im_before, xs, im_watermark, sl, int(label), xuhao, verbose=False)

    class MyTakeStep(object):
        def __init__(self, stepsize=5):
            self.stepsize = stepsize * mutation_coeff

        def __call__(self, x):
            x[0] += np.random.uniform(-1 * self.stepsize, 1 * self.stepsize)
            x[1] += np.random.uniform(-3 * self.stepsize, 3 * self.stepsize)
            x[2] += np.random.uniform(-3 * self.stepsize, 3 * self.stepsize)
            x[3] += np.random.uniform(-1 * self.stepsize, 1 * self.stepsize)
            x[4] += np.random.uniform(-1 * self.stepsize, 1 * self.stepsize)
            x[5] += np.random.uniform(-0.5 * self.stepsize, 0.5 * self.stepsize)
            x[6] += np.random.uniform(-0.5 * self.stepsize, 0.5 * self.stepsize)
            x[7] += np.random.uniform(-0.5 * self.stepsize, 0.5 * self.stepsize)
            x[8] += np.random.uniform(-0.5 * self.stepsize, 0.5 * self.stepsize)
            x[9] += np.random.uniform(-2 * self.stepsize, 2 * self.stepsize)
            return x

    mytakestep = MyTakeStep()

    class MyBounds(object):
        def __init__(self):
            self.bounds = [
                [0, 15],
                [0, 224 - watermark_x1],
                [0, 224 - watermark_y1],
                [0, 99],
                [0, 99],
                [2, 5],
                [2, 5],
                [0, 10],
                [0, 10],
                [17, 600]
            ]

        def __call__(self,** kwargs):
            x = kwargs["x_new"]
            return all(self.bounds[i][0] <= x[i] <= self.bounds[i][1] for i in range(10))

    mybounds = MyBounds()

    attack_result = basinhopping(
        func=predict_fn,
        x0=x0,
        callback=callback_fn,
        take_step=mytakestep,
        accept_test=mybounds,
        niter=N_ITER
    )
    return attack_result.x, attack_result.fun


# 进化策略
def evolutionary_attack(im_before, label, im_watermark, sl, xuhao):
    original_wm_width, original_wm_height = im_watermark.size
    ref_threshold = (original_wm_width * original_wm_height) // 2
    current_w, current_h = original_wm_width, original_wm_height

    if current_w * current_h > ref_threshold:
        adjusted_w = original_wm_width // 2
        adjusted_h = original_wm_height // 2
    else:
        adjusted_w, adjusted_h = current_w, current_h

    watermark_scale = min(224 / (sl * adjusted_w), 224 / (sl * adjusted_h))
    watermark_x1 = int(adjusted_w * watermark_scale)
    watermark_y1 = int(adjusted_h * watermark_scale)

    param_ranges = [
        [0, 15],
        [0, 224 - watermark_x1],
        [0, 224 - watermark_y1],
        [0, 99],
        [0, 99],
        [2, 5],
        [2, 5],
        [0, 10],
        [0, 10],
        [17, 600]
    ]

    # 初始化种群
    def initialize_population(size):
        population = []
        for _ in range(size):
            individual = []
            for i in range(10):
                individual.append(np.random.uniform(param_ranges[i][0], param_ranges[i][1]))
            population.append(individual)
        return population

    # 交叉操作
    def crossover(population, mutated):
        offspring = []
        for i in range(len(population)):
            child = []
            for j in range(10):
                if np.random.rand() < CR or j == np.random.randint(10):
                    child.append(mutated[i][j])
                else:
                    child.append(population[i][j])
            offspring.append(child)
        return offspring

    # 变异操作
    def mutate(population, mutation_coeff):
        mutated = []
        for individual in population:
            new_ind, _ = bh_optimization(im_before, label, im_watermark, sl, xuhao, individual, mutation_coeff)
            mutated.append(new_ind)
        return mutated

    # 选择操作
    def select(population, offspring):
        combined = population + offspring
        scores = []
        for ind in combined:
            score = predict_classes(im_before, ind, im_watermark, int(label), sl, xuhao)
            scores.append(score)
        selected_indices = np.argsort(scores)[:NP]
        return [combined[i] for i in selected_indices]

    population = initialize_population(NP)

    for ind in population:
        if attack_success(im_before, ind, im_watermark, sl, int(label), xuhao):
            return ind, True

    for gen in range(GENERATIONS):
        mutation_coeff = MUTATION_COEFF_INIT - (MUTATION_COEFF_INIT - MUTATION_COEFF_FINAL) * (gen / GENERATIONS)
        print(f"进化代数: {gen + 1}/{GENERATIONS}，当前变异系数: {mutation_coeff:.2f}")

        mutated = mutate(population, mutation_coeff)
        offspring = crossover(population, mutated)
        population = select(population, offspring)

        for ind in population:
            if attack_success(im_before, ind, im_watermark, sl, int(label), xuhao):
                return ind, True

    best_ind = min(population, key=lambda x: predict_classes(im_before, x, im_watermark, int(label), sl, xuhao))
    success = attack_success(im_before, best_ind, im_watermark, sl, int(label), xuhao)
    return best_ind, success


if __name__ == "__main__":
    true_list = []
    count = 0
    success_count = 0

    output_dir = f'.\\scale_{scale_size}\\{watermark_path}'
    os.makedirs(output_dir, exist_ok=True)

    for dir_name in ['.\\output', '.\\pic']:
        os.makedirs(dir_name, exist_ok=True)

    for i in range(2, 10):
        try:

            path = f'.\\Raw Data\\{i}.png'
            if not os.path.exists(path):
                print(f"图像 {i} 不存在: {path}")
                continue
            img = Image.open(path).convert("RGB")

            predict = label_model(img).cpu().detach().numpy()
            original_class = np.argmax(predict)
            target_class = int(classes[i])

            if original_class == target_class:
                print(f"\n***** 处理图像 {i} *****")
                true_list.append(i)

                best_params, success = evolutionary_attack(img, classes[i], im_watermark, scale_size, xuhao=i)

                result_img = Image.open('./temp1_VGG.png').convert('RGB')
                new_predict = label_model(result_img).cpu().detach().numpy()
                new_class = np.argmax(new_predict)

                if success:
                    success_count += 1
                    print(f"图像 {i} 攻击成功")
                else:
                    count += 1
                    print(f"图像 {i} 攻击失败")

                result_img.save(f'{output_dir}\\advwatermark_{i}.png')

        except Exception as e:
            print(f"处理图像 {i} 出错: {str(e)}")

    print("\n===== 攻击结果 =====")
    if len(true_list) == 0:
        print("攻击成功率: 0.00%")
    else:
        print(f"攻击成功率: {(success_count / len(true_list)) * 100:.2f}%")