import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def length_reward(x, p=0.5):
    """
    计算长度奖励值
    
    参数:
    x -- 归一化后的响应长度 (L/L_max)
    p -- 分段函数的转折点，默认值为0.5
    
    返回:
    长度奖励值
    """
    if 0 <= x <= p:
        return 1 - (1 - x/p) **2
    elif p < x <= 1:
        return 1 - 2 * ((x - p) / (1 - p))** 2
    else:
        return None  # 超出[0,1]范围时返回None

# 生成x值（归一化后的响应长度）
x_vals = np.linspace(0, 1, 1000)

# 计算对应的奖励值
rewards = [length_reward(x, p=0.5) for x in x_vals]

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x_vals, rewards, 'b-', linewidth=2)

# 标记转折点p=0.5
p = 0.5
plt.plot(p, length_reward(p), 'ro', markersize=8)
plt.text(p + 0.02, length_reward(p) + 0.05, f'p={p}, R(p)={length_reward(p):.2f}', fontsize=12)

# 装饰图形
plt.title('长度奖励函数图像', fontsize=16)
plt.xlabel('归一化响应长度 x = L/L_max', fontsize=14)
plt.ylabel('奖励值 R_l', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1)
plt.ylim(0, 1.1)  # 稍微扩展y轴范围，使图像更清晰

# 添加注释说明函数分段
plt.text(0.25, 0.8, r'$R_l = 1 - (1 - \frac{x}{p})^2$', fontsize=12)
plt.text(0.7, 0.8, r'$R_l = 1 - 2(\frac{x-p}{1-p})^2$', fontsize=12)

plt.tight_layout()
plt.savefig('length_reward_function.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制不同p值的对比图
plt.figure(figsize=(12, 8))
p_values = [0.3, 0.5, 0.7]
colors = ['r', 'g', 'b']

for p, color in zip(p_values, colors):
    rewards = [length_reward(x, p=p) for x in x_vals]
    plt.plot(x_vals, rewards, color=color, linewidth=2, label=f'p={p}')
    plt.plot(p, length_reward(p, p=p), f'{color}o', markersize=6)

# 装饰图形
plt.title('不同p值的长度奖励函数对比', fontsize=16)
plt.xlabel('归一化响应长度 x = L/L_max', fontsize=14)
plt.ylabel('奖励值 R_l', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('length_reward_comparison.png', dpi=300, bbox_inches='tight')
plt.show()