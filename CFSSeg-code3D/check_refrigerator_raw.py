import os
import glob
import numpy as np

def check_refrigerator_raw():
    """检查过滤前原始处理数据中的refrigerator类（索引17）"""

    # 数据路径
    data_path = 'datasets/ScanNet/blocks_bs1_s1/data'
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据目录 {data_path}")
        return

    # 加载scannet类名
    classes_file = 'datasets/ScanNet/meta/scannet_classnames.txt'
    if not os.path.exists(classes_file):
        print(f"错误: 找不到类名文件 {classes_file}")
        return

    class_names = [line.strip() for line in open(classes_file, 'r').readlines()]
    refrigerator_id = class_names.index('refrigerator') if 'refrigerator' in class_names else -1

    if refrigerator_id == -1:
        print("错误: 类名文件中没有refrigerator类")
        return

    print(f"refrigerator类的索引为: {refrigerator_id}")

    # 查找所有数据文件
    data_files = glob.glob(os.path.join(data_path, "*.npy"))
    print(f"找到 {len(data_files)} 个数据文件")

    # 统计包含refrigerator类的场景
    refrigerator_scenes = []
    refrigerator_points_total = 0
    total_points = 0

    for file_path in data_files:
        try:
            data = np.load(file_path)
            labels = data[:, 6].astype(np.int64)
            scene_name = os.path.basename(file_path)[:-4]

            # 检查是否包含refrigerator类
            refrigerator_points = np.sum(labels == refrigerator_id)
            if refrigerator_points > 0:
                total_scene_points = len(labels)
                percentage = refrigerator_points / total_scene_points * 100

                refrigerator_scenes.append({
                    'scene': scene_name,
                    'points': refrigerator_points,
                    'total_points': total_scene_points,
                    'percentage': percentage,
                    'threshold': max(int(total_scene_points * 0.05), 100)
                })

                refrigerator_points_total += refrigerator_points
                total_points += total_scene_points

                print(f"\n发现refrigerator在场景 {scene_name}:")
                print(f"  - refrigerator点数: {refrigerator_points}")
                print(f"  - 总点数: {total_scene_points}")
                print(f"  - 占比: {percentage:.2f}%")
                print(f"  - 过滤阈值: {max(int(total_scene_points * 0.05), 100)}")
                print(f"  - 过滤结果: {'保留' if refrigerator_points > max(int(total_scene_points * 0.05), 100) else '过滤'}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 输出统计结果
    print("\n===== 统计结果 =====")
    print(f"包含refrigerator的场景数量: {len(refrigerator_scenes)}")
    print(f"refrigerator点总数: {refrigerator_points_total}")
    print(f"总点数: {total_points}")
    print(f"总体占比: {refrigerator_points_total / total_points * 100 if total_points > 0 else 0:.2f}%")

    if refrigerator_scenes:
        # 按照点数排序
        refrigerator_scenes.sort(key=lambda x: x['points'], reverse=True)

        print("\n按点数排序的场景:")
        for i, scene in enumerate(refrigerator_scenes):
            print(f"{i+1}. {scene['scene']}: {scene['points']} 点 ({scene['percentage']:.2f}%)")

        # 检查过滤后剩余的场景
        filtered_scenes = [scene for scene in refrigerator_scenes if scene['points'] > scene['threshold']]
        print(f"\n过滤后剩余场景数量: {len(filtered_scenes)}")
        if filtered_scenes:
            print("\n过滤后的场景:")
            for i, scene in enumerate(filtered_scenes):
                print(f"{i+1}. {scene['scene']}: {scene['points']} 点 ({scene['percentage']:.2f}%)")

if __name__ == "__main__":
    check_refrigerator_raw()