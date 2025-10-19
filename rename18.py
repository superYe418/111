import os

def rename_folders_and_files(base_path):
    """
    遍历base_path目录下的所有文件夹，将文件夹重命名为BraTS18_Training_x，
    同时将文件夹内所有文件名中的原文件夹名替换为新文件夹名。
    
    :param base_path: 父目录路径
    """
    try:
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        subfolders.sort()  # 按名称排序，便于一致性
        
        for index, folder in enumerate(subfolders, start=1):
            old_folder_path = os.path.join(base_path, folder)
            new_folder_name = f"BraTS18_Training_{index}"
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            # 重命名文件夹
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed folder: {old_folder_path} -> {new_folder_path}")
            
            # 遍历该文件夹中的所有文件
            for file_name in os.listdir(new_folder_path):
                old_file_path = os.path.join(new_folder_path, file_name)
                
                if os.path.isfile(old_file_path):
                    # 替换文件名中的旧文件夹名为新文件夹名
                    new_file_name = file_name.replace(folder, new_folder_name)
                    new_file_path = os.path.join(new_folder_path, new_file_name)
                    
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} -> {new_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
base_directory = "/Code/SFusion-main/dataset/MICCAI_BraTS_2018_Data_Training/HGG"
rename_folders_and_files(base_directory)




def rename_folders_and_files(base_path):
    """
    遍历base_path目录下的所有文件夹，将文件夹重命名为BraTS18_Training_x，
    同时将文件夹内所有文件名中的原文件夹名替换为新文件夹名。
    
    :param base_path: 父目录路径
    """
    try:
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        subfolders.sort()  # 按名称排序，便于一致性
        
        for index, folder in enumerate(subfolders, start=1):
            old_folder_path = os.path.join(base_path, folder)
            new_folder_name = f"BraTS18_Training_{index}"
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            # 重命名文件夹
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed folder: {old_folder_path} -> {new_folder_path}")
            
            # 遍历该文件夹中的所有文件
            for file_name in os.listdir(new_folder_path):
                old_file_path = os.path.join(new_folder_path, file_name)
                
                if os.path.isfile(old_file_path):
                    # 替换文件名中的旧文件夹名为新文件夹名
                    new_file_name = file_name.replace(folder, new_folder_name)
                    new_file_path = os.path.join(new_folder_path, new_file_name)
                    
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} -> {new_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
base_directory = "/Code/SFusion-main/dataset/MICCAI_BraTS_2018_Data_Training/HGG"
rename_folders_and_files(base_directory)
