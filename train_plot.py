import matplotlib.pyplot as plt

# 設定圖片和標題的列表
# image0 = 'result/train/Loss_epoch10.png'
image1 = 'result/train/Loss_epoch100.png'
image2 = 'result/train/Loss_epoch200.png'
image3 = 'result/train/Loss_epoch300.png'
image4 = 'result/train/Loss_epoch400.png'
image5 = 'result/train/Loss_epoch500.png'
image6 = 'result/train/Loss_epoch600.png'
image7 = 'result/train/Loss_epoch700.png'
image8 = 'result/train/Loss_epoch800.png'
# image0 = 'result/train/acc_epoch10.png'
# image1 = 'result/train/acc_epoch100.png'
# image2 = 'result/train/acc_epoch200.png'
# image3 = 'result/train/acc_epoch300.png'
# image4 = 'result/train/acc_epoch400.png'
# image5 = 'result/train/acc_epoch500.png'
# image6 = 'result/train/acc_epoch600.png'
# image7 = 'result/train/acc_epoch700.png'
# image8 = 'result/train/acc_epoch800.png'

# images = [image0, image1, image2, image3, image4, image5, image6, image7]
images = [image1, image2, image3, image4, image5, image6, image7, image8]
# titles = ['Epoch=10', 'Epoch=100', 'Epoch=200', 'Epoch=300', 'Epoch=400', 'Epoch=500', 'Epoch=600', 'Epoch=700']
titles = ['Epoch=100', 'Epoch=200', 'Epoch=300', 'Epoch=400', 'Epoch=500', 'Epoch=600', 'Epoch=700', 'Epoch=800']

plt.figure(dpi=1000)
# 創建一個2x4的子圖表格
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# 將圖片和標題逐一添加到子圖中
for i, ax in enumerate(axes.flat):
    # 讀取圖片並顯示
    img = plt.imread(images[i])
    ax.imshow(img)
    # 設定標題
    ax.set_title(titles[i])
    # 隱藏軸刻度
    ax.axis('off')

# 調整子圖之間的間距
plt.tight_layout()

plt.savefig('result/train/loss.png')
# plt.savefig('result/train/acc.png')
# 顯示圖片
plt.show()
