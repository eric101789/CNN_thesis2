"""
程式碼功能:
讀取測試集內的所有PNG檔案
將PNG圖片路徑轉換成DataFrame
將DataFrame寫入至CSV檔案內
"""

import os
import glob
import pandas as pd

data_dir = 'dataset/'  # 設定圖片所在的資料夾路徑
image_paths = glob.glob(os.path.join(data_dir, '*.png'))  # 收集所有.png檔案的路徑

# 創建包含圖片路徑的DataFrame
df = pd.DataFrame({'path': image_paths})
df.to_csv('dataset.csv', index=False)  # 將DataFrame寫入CSV檔案
