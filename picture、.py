import numpy as np
import matplotlib.pyplot as plt

# 假设数据：SNR 和 PSNR（dB）
SNR = np.array([0, 5, 10, 15, 20, 25, 30, 35])  # 示例的SNR值
PSNR_mambaJSCC = np.array([10, 15, 20, 22, 24, 25, 26, 27])  # 假设的mambaJSCC PSNR值
PSNR_BPG_LDPC = np.array([12, 16, 18, 20, 21, 22, 23, 23.5])  # 假设的BPG + LDPC PSNR值
PSNR_DeepJSCC = np.array([8, 13, 17, 19, 21, 23, 24, 24.5])  # 假设的DeepJSCC PSNR值

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制不同的曲线
plt.plot(SNR, PSNR_mambaJSCC, marker='o', color='blue', label='mambaJSCC w/ SA&RA', linestyle='-', markersize=6)
plt.plot(SNR, PSNR_BPG_LDPC, marker='s', color='green', label='BPG + LDPC', linestyle='--', markersize=6)
plt.plot(SNR, PSNR_DeepJSCC, marker='^', color='red', label='DeepJSCC', linestyle=':', markersize=6)

# 设置标题和标签
plt.title('Performance Comparison under Different SNRs', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)

# 添加图例
plt.legend(loc='lower right', fontsize=12)

# 设置网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
