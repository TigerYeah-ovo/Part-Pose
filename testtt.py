import cv2
import numpy as np
 
# 1. 读取图像并灰度化
img = cv2.imread('/home/yecl24/workspace/6DoFPoseEstimation/part_registration/code/AG-Pose-main/figure/cup.jpg')  # 读取彩色图像
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图像
 
# 2. 调用Harris角点检测函数
# 参数说明：
# - gray：输入灰度图像
# - blockSize：角点检测的邻域大小（通常取3~5）
# - ksize：Sobel算子的窗口大小（必须为奇数，通常取3）
# - k：经验参数（0.04~0.06）
dst = cv2.cornerHarris(gray, blockSize=4, ksize=3, k=0.04)
 
# 3. 阈值筛选并标记角点（红色：BGR格式为[0,0,255]）
img[dst > 0.05 * dst.max()] = [0, 0, 255]  # 响应值大于阈值的像素标记为红色
 
# 4. 显示结果
cv2.imshow('Harris Corner Detection', img)
cv2.waitKey(0)  # 等待按键关闭窗口