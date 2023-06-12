import numpy as np
from scipy.spatial import distance
from scipy.stats import matrix_normal
import cv2
import os


def predict_util(file_path ):
    noisy_img = main(file_path,eps = 0.1)
    cv2.imwrite('noisy.png',noisy_img)
    return noisy_img



def coder_distance(eps, x, z):
    hist_x, _ = np.histogram(x, bins=1024)#np.histogram()是一个生成直方图的函数 待统计数组为x,统计的区间数为1024
    hist_z, _ = np.histogram(z, bins=1024)
    d = np.mean((x - z) ** 2)/(np.prod(x.shape)) + distance.jensenshannon(hist_x, hist_z)
    #mean函数用于求取平均值 prod求连乘后的结果
    return np.exp(-eps * d)
def add_noise(img,eps):

    distra=matrix_normal(mean=img)

    while True:
        z = np.random.normal(0, 5/eps, size=img.shape)
        #从正态（高斯）分布中抽取随机样本 分布的均值（中心）：0 分布的标准差（宽度）：5/eps 输出值的维度：img.shape
        noisy_img = img + z
    
        u = np.random.uniform(0, 1)
        #作用于从一个均匀分布的区域中随机采样 u取一个随机数
        ratio = coder_distance(eps, img, noisy_img) / ( 2.5 * distra.pdf(noisy_img))

        if u <= ratio:
            noisy_img = noisy_img
            break

    return noisy_img


def gray_to_color(src,src_gray):

    B = src[:,:,0]
    G = src[:,:,1]
    R = src[:,:,2]
    g = src_gray[:]
    p = 0.2989; q = 0.5870; t = 0.1140
    B_new = (g-p*R-q*G)/t
    B_new = np.uint8(B_new)#将当前数组作为图像类型来进行操作
    src_new = np.zeros((src.shape)).astype("uint8") # zeros(shape, dtype=float, order=‘C’)
    src_new[:,:,0] = B_new
    src_new[:,:,1] = G
    src_new[:,:,2] = R

    return src_new

def main(file_path,eps):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # cv2.imread(filename, flags) cv2.imread(filename, flags) 参数： filepath：读入imge的完整路径 flags：标志位 unchanded 读入完整图像
    if len(img.shape) == 2:
    #如果是灰度图像
        return add_noise(img,eps)
    
    else:
    #如果是彩色图像   
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gray_noise = add_noise(img_gray,eps)
        # return gray_to_color(img,img_gray_noise)

        arr1 = img[:, :, 0]
        arr2 = img[:, :, 1]
        arr3 = img[:, :, 2]

        arr1_noise = add_noise(arr1,eps)
        arr2_noise = add_noise(arr2,eps)
        arr3_noise = add_noise(arr3,eps)

        return np.concatenate((arr1_noise[:, :, np.newaxis], arr2_noise[:, :, np.newaxis], arr3_noise[:, :, np.newaxis]), axis=2)
        #能够一次完成多个数组的拼接。

print(os.path.dirname(os.path.abspath(__file__)))

# if __name__ == "__main__":
#     predict_util()
#     file_path = '2.jpg'

#     noisy_img = main(file_path,eps = 0.1)

#     cv2.imwrite('noisy.png',noisy_img)
