#!/usr/bin/env python3
"""
批量缩放 URDF 文件中的 mesh 模型
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]

def scale_urdf_file(urdf_path, scale_factor, backup=True):
    """
    缩放 URDF 文件中的所有 mesh

    Args:
        urdf_path: URDF 文件路径
        scale_factor: 缩放因子 (0.3 表示缩放到 30%)
        backup: 是否备份原文件
    """
    urdf_path = Path(urdf_path)

    if backup:
        backup_path = urdf_path.with_suffix('.urdf.bak')
        shutil.copy(urdf_path, backup_path)
        print(f"已备份: {backup_path}")

    # 读取并解析 URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 查找所有 mesh 标签并添加/修改 scale 属性
    count = 0
    for mesh in root.findall('.//mesh'):
        current_scale = mesh.get('scale', '1 1 1')
        sx, sy, sz = [float(x) * scale_factor for x in current_scale.split()]
        mesh.set('scale', f'{sx} {sy} {sz}')
        count += 1

    # 保存修改后的文件
    tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"已缩放 {urdf_path}, 修改了 {count} 个 mesh")

    return count

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  缩放单个文件: python scale_urdf.py <urdf_file> <scale_factor>")
        print("  缩放所有URDF: python scale_urdf.py --all <scale_factor>")
        print("\n示例:")
        print("  python scale_urdf.py 100466/mobility.urdf 0.3")
        print("  python scale_urdf.py --all 0.3")
        sys.exit(1)

    scale_factor = float(sys.argv[-1])

    if sys.argv[1] == "--all":
        # 缩放所有 URDF 文件
        urdf_files = list(ROOT_DIR.rglob("*.urdf"))
        total = 0
        for urdf_file in urdf_files:
            total += scale_urdf_file(urdf_file, scale_factor)
        print(f"\n总共修改了 {total} 个 mesh")
    else:
        # 缩放单个文件
        scale_urdf_file(sys.argv[1], scale_factor)
