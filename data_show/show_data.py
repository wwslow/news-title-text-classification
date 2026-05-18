import pandas as pd
import matplotlib.pyplot as plt
import jieba
from rf.config import Config
config=Config()
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv(config.train_csv_path, encoding='utf-8')

# 创建画布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 样本类别分布（柱状图）
ax1 = axes[0, 0]
label_counts = df['label'].value_counts()
colors = plt.cm.Set3(range(len(label_counts)))
bars = ax1.barh(range(len(label_counts)), label_counts.values, color=colors)
ax1.set_yticks(range(len(label_counts)))
ax1.set_yticklabels(label_counts.index, fontsize=9)
ax1.set_xlabel('样本数量', fontsize=11)
ax1.set_title('样本类别分布', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# 在柱子上添加数值标签
for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
    ax1.text(bar.get_width() + max(label_counts.values) * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f'{count}', va='center', fontsize=8)

# 2. 样本类别分布（饼图）
ax2 = axes[0, 1]
sizes = label_counts.values
labels_pie = [f'{label}\n({count})' for label, count in zip(label_counts.index, sizes)]
wedges, texts, autotexts = ax2.pie(sizes, labels=labels_pie, autopct='', startangle=90,
                                   textprops={'fontsize': 7})
ax2.set_title('类别占比分布', fontsize=13, fontweight='bold')

# 3. 句子长度分布（直方图）
ax3 = axes[1, 0]
# 计算每个句子的字符数
char_lengths = df['text'].apply(lambda x: len(str(x)))
ax3.hist(char_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(char_lengths.mean(), color='red', linestyle='--', linewidth=2,
            label=f'平均值: {char_lengths.mean():.1f}')
ax3.axvline(char_lengths.median(), color='green', linestyle='--', linewidth=2,
            label=f'中位数: {char_lengths.median():.1f}')
ax3.set_xlabel('句子长度（字符数）', fontsize=11)
ax3.set_ylabel('频数', fontsize=11)
ax3.set_title('句子长度分布（字符）', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)

# 4. 分词后的词数分布
ax4 = axes[1, 1]
word_lengths = df['text'].apply(lambda x: len(jieba.lcut(str(x))))
ax4.hist(word_lengths, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax4.axvline(word_lengths.mean(), color='red', linestyle='--', linewidth=2,
            label=f'平均值: {word_lengths.mean():.1f}')
ax4.axvline(word_lengths.median(), color='green', linestyle='--', linewidth=2,
            label=f'中位数: {word_lengths.median():.1f}')
ax4.set_xlabel('句子长度（词数）', fontsize=11)
ax4.set_ylabel('频数', fontsize=11)
ax4.set_title('句子长度分布（分词后）', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig('data/data_distribution.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存到: data/data_distribution.png")
plt.show()