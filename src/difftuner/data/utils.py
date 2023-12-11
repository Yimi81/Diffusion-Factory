import os
import pandas as pd

# 文件夹路径
folder_path = '/mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/hyhh_dataset'

# 用于存储文件名和内容的列表
file_contents = []

# 遍历文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            file_extension = os.path.splitext(filename)[1]
            file_contents.append({'file_name': filename.replace(file_extension, '.png'), 'text': content})

# 将列表转换为DataFrame
df = pd.DataFrame(file_contents)

# 将DataFrame保存为JSON Lines文件
output_file = '/mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/hyhh_dataset/metadata.jsonl'
df.to_json(output_file, orient='records', lines=True)
