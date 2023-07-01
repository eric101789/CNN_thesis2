import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result/test/test_LSTM_results_epoch200.csv')

plt.figure(dpi=300)
plt.hist(df['class_probability'], bins=10)
plt.xlabel('Probability')
plt.ylabel('Frame')
plt.title('Probability Distribution of Test LSTM Results')
plt.savefig('result/test/test_LSTM_probability_epoch200.png')
plt.show()


# 繪製折線圖
plt.figure(dpi=300, figsize=(70, 6.5))
plt.plot(df['class_probability'])
# 設置標籤和標題
plt.xlabel('Sample')
plt.ylabel('Probability')
plt.title('Prediction Probability')
plt.savefig('result/test/test_LSTM_probability_plot_epoch200.png')
plt.show()

