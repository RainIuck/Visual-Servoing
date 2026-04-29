from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
asset_dir = ROOT_DIR / "asset"
table_config = {
    "urdf_path": str(asset_dir / "scene/table/table.urdf"),
    "object_name": "table",
    "position": [0, 0, 0],
    "orientation": [1, 0, 0, 0]
}

banana_config = {
    "object_collision_meshes": str(asset_dir / "object/banana/collision.obj"),
    "object_visual_meshes": str(asset_dir / "object/banana/visual.glb"),
    "object_name": "banana"
}
cup_config = {
    "object_collision_meshes": str(asset_dir / "object/cup/textured.obj"),
    "object_visual_meshes": str(asset_dir / "object/cup/base.glb"),
    "object_name": "cup",
    "position": [0.5, 0, 0],
    "orientation": [1, 0, 0, 0]
}

bottle_config = {
    "urdf_path": str(asset_dir / "object/3517/mobility.urdf"),
    "object_name": "bottle",
    "position": [0.5, 0, 0],
    "orientation": [1, 0, 0, 0],
    
}
drawer_config = {
    "urdf_path": str(asset_dir / "object/19179/mobility.urdf"),
    "object_name": "drawer",
    "position": [1.5, 0, 0],
    "orientation": [1, 0, 0, 0]
}


box_config = {
    # 确保这里的路径与你实际放置 100466 文件夹的位置一致
    "urdf_path": str(ROOT_DIR / "100466/mobility.urdf"),
    "object_name": "box",
    "position": [0.6, 0.3, 0], # 把它放在机械臂右前方的位置
    "orientation": [1, 0, 0, 0],
   
}

block_config = {
    # 路径指向你刚建的方块
    "urdf_path": str(asset_dir / "object/block.urdf"), 
    "object_name": "block",
    # 尺寸是 0.04，所以高度 Z 设为 0.02 刚好贴地
    "position": [0.4, 0.0, 0.02], 
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.02,
}

block_red_config = {
    "urdf_path": str(asset_dir / "object/block_red.urdf"),
    "object_name": "block_red",
    "position": [0.4, 0.2, 0.02],  # 左边
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.02,
}

block_green_config = {
    "urdf_path": str(asset_dir / "object/block_green.urdf"),
    "object_name": "block_green",
    "position": [0.4, 0.0, 0.02],  # 中间
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.02,
}

block_yellow_config = {
    "urdf_path": str(asset_dir / "object/block_yellow.urdf"),
    "object_name": "block_yellow",
    "position": [0.4, -0.2, 0.02], # 右边
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.02,
}

cuboid_orange_config = {
    "urdf_path": str(asset_dir / "object/cuboid_orange.urdf"),
    "object_name": "cuboid_orange",
    "position": [0.4, 0.0, 0.016],
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.016,
}

sphere_purple_config = {
    "urdf_path": str(asset_dir / "object/sphere_purple.urdf"),
    "object_name": "sphere_purple",
    "position": [0.4, 0.0, 0.025],
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.025,
}

cylinder_cyan_config = {
    "urdf_path": str(asset_dir / "object/cylinder_cyan.urdf"),
    "object_name": "cylinder_cyan",
    "position": [0.4, 0.0, 0.023],
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.023,
}

arch_magenta_config = {
    "urdf_path": str(asset_dir / "object/arch_magenta.urdf"),
    "object_name": "arch_magenta",
    "position": [0.4, 0.0, 0.025],
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.025,
}

l_shape_gray_config = {
    "urdf_path": str(asset_dir / "object/l_shape_gray.urdf"),
    "object_name": "l_shape_gray",
    "position": [0.4, 0.0, 0.015],
    "orientation": [1, 0, 0, 0],
    "spawn_z": 0.015,
}
