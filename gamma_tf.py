import cv2
import numpy as np
import matplotlib.pyplot as plt

#分道计算每个通道的直方图
img0 = cv2.imread('image1.jpeg')
hist_b = cv2.calcHist([img0],[0],None,[256],[0,256])
hist_g = cv2.calcHist([img0],[1],None,[256],[0,256])
hist_r = cv2.calcHist([img0],[2],None,[256],[0,256])
def gamma_trans(img,gamma):
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #实现映射用的是Opencv的查表函数
    return cv2.LUT(img0,gamma_table)
img0_corrted = gamma_trans(img0, 0.5)
cv2.imshow('img0',img0)
cv2.imshow('gamma_image',img0_corrted)
cv2.imwrite('gamma_image.png',img0_corrted)
#分通道计算Gamma校正后的直方图
hist_b_c =cv2.calcHist([img0_corrted],[0],None,[256],[0,256])
hist_g_c =cv2.calcHist([img0_corrted],[1],None,[256],[0,256])
hist_r_c =cv2.calcHist([img0_corrted],[2],None,[256],[0,256])
fig = plt.figure('gamma')
pix_hists = [[hist_b, hist_g, hist_r],
    [hist_b_c, hist_g_c, hist_r_c]]
pix_vals = range(256)
for sub_plt, pix_hist in zip([121, 122], pix_hists):
    ax = fig.add_subplot(sub_plt, projection='3d')
    for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
          cs = [c] * 256
          ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='none', lw=0)
    ax.set_xlabel('Pixel Values')
    ax.set_xlim([0, 256])
    ax.set_ylabel('Count')
    ax.set_zlabel('Channels')
plt.show() 
cv2.waitKey()
