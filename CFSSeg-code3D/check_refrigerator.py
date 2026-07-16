import os
import pickle
import numpy as np

def check_dataset():
    # 加载class2scans.pkl文件
    class2scans_file = 'datasets/ScanNet/blocks_bs1_s1/class2scans.pkl'
    if not os.path.exists(class2scans_file):
        print(f"错误：找不到文件 {class2scans_file}")
        return

    with open(class2scans_file, 'rb') as f:
        class2scans = pickle.load(f)

    # 检查所有类别的场景数量
    print("所有类别的场景数量:")
    for class_id, scenes in class2scans.items():
        print(f"类别 {class_id}: {len(scenes)} 个场景")

    # 检查数据目录中的文件
    data_dir = 'datasets/ScanNet/blocks_bs1_s1/data'
    if os.path.exists(data_dir):
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        print(f"\n数据目录中的.npy文件数量: {len(npy_files)}")

        # 随机检查几个文件中的标签分布
        print("\n随机检查几个文件中的标签分布:")
        for i in range(min(5, len(npy_files))):
            file = npy_files[i]
            data = np.load(os.path.join(data_dir, file))
            labels = data[:, 6].astype(np.int64)
            unique_labels = np.unique(labels)
            print(f"\n文件 {file}:")
            print(f"总点数: {len(labels)}")
            print(f"包含的标签: {unique_labels}")
            for label in unique_labels:
                count = np.sum(labels == label)
                print(f"标签 {label}: {count} 个点 ({count/len(labels)*100:.2f}%)")
    else:
        print(f"\n错误：找不到数据目录 {data_dir}")

if __name__ == '__main__':
    check_dataset()