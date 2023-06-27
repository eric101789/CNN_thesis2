"""
程式碼功能
    1. 將數據子載波( len = 52 )置於一個 list 的前面，然後在後面補 0 直到 len = 64
    2. 將補 0 後的數據子載波轉換成矩陣
    3. 將矩陣轉換成圖片並儲存，另外還可以等比例放大圖片
參考 :
    補 0 : https://blog.csdn.net/Hodors/article/details/117438511
    list 轉成矩陣 : https://www.delftstack.com/zh-tw/howto/python/list-to-matrix-python/#%e5%9c%a8-python-%e4%b8%ad%e4%bd%bf%e7%94%a8%e8%bf%b4%e5%9c%88%e5%92%8c%e5%88%97%e8%a1%a8%e5%88%87%e7%89%87%e5%b0%87%e5%88%97%e8%a1%a8%e8%bd%89%e6%8f%9b%e7%82%ba%e9%99%a3%e5%88%97%e6%88%96%e7%9f%a9%e9%99%a3
    等比例放大圖片 : https://www.zhihu.com/question/311903523
"""
import csv
import re
from math import sqrt, atan2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """
    FILE_NAME1 = "training.csv"                    # 原始 csv 檔案
    PATH = "dataset/"

    f1 = open(FILE_NAME1)

    loop1 = 0
    loop_n = 11000                           # 設定總共要輸出多少筆資料

    for j1, l1 in enumerate(f1.readlines()):
        imaginary1 = []
        real1 = []
        amplitudes1 = []
        phases1 = []

        # Parse string to create integer list
        csi_string1 = re.findall(r"\[(.*)\]", l1)[0]
        csi_raw1 = [int(x) for x in csi_string1.split(" ") if x != '']

        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw1)):
            if i % 2 == 0:
                imaginary1.append(csi_raw1[i])
            else:
                real1.append(csi_raw1[i])

        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw1) / 2)):
            # 把計算後的振幅值取 3 位小數點後，放入 list 裡
            amplitudes1.append(format(sqrt(imaginary1[i] ** 2 + real1[i] ** 2), '.3f'))
            phases1.append(format(atan2(imaginary1[i], real1[i]), '.3f'))

        if loop1 >= loop_n:
            break

        data_sub = []
        # 將數據子載波的上、下兩部分整合成一個新的 list
        # data_sub = [element for e in [amplitudes1[6:32], amplitudes1[33:59]] for element in e]
        data_sub = amplitudes1[6:32] + amplitudes1[33:59]
        # print('數據子載波 : ', data_sub)

        data_sub.extend(0 for _ in range(abs(64-len(data_sub))))
        # print('補0後的數據子載波 : ', data_sub)

        matrix_data_sub = list(np.array_split(data_sub, 8))
        matrix = np.array(matrix_data_sub)
        # print('數據子載波的矩陣 : \n', matrix)

        size = (8, 8)
        img = Image.new('L', size)
        pixels = img.load()
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = int(float(matrix[j][i]))

        # 等比例放大圖片
        w, h = img.size
        # print("原圖片的大小 : ", img.size)
        width = img.size[0]   # 圖片寬度
        height = img.size[1]   # 圖片高度
        scale = 8.0
        reimg = img.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
        # print("放大後圖片的大小 : ", reimg.size)

        # 儲存圖片
        address = PATH + "photo-{}.png".format(j1)   # 編輯圖片的名稱
        reimg.save(address)                          # 儲存圖片

        loop1 += 1

    print("轉換完成!")




