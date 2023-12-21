import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepocess dataset")
    parser.add_argument(
        "--folder_path",
        type=str,
        default=None,
        required=True,
        help="The folder where the pending dataset is located."
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default=".jpg",
        help="Format of image to be processed."
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # 用于存储文件名和内容的列表
    file_contents = []
    file_list = os.listdir(args.folder_path)
    file_list.sort()

    # 将所有不是图像和文本的文件删除
    for filename in file_list:
        if not filename.endswith('.txt') and not filename.endswith(args.image_format):
            file_path = os.path.join(args.folder_path, filename)
            os.remove(file_path)

    # 构建Metadata
    for index, filename in enumerate(file_list, start=1):
        if filename.endswith('.txt'):
            file_path = os.path.join(args.folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_extension = os.path.splitext(filename)[1]
                file_contents.append({'file_name': filename.replace(file_extension, '.png'), 'text': content})
    # 将列表转换为DataFrame
    df = pd.DataFrame(file_contents)

    # 将DataFrame保存为JSON Lines文件
    output_file = os.path.join(args.folder_path, "metadata.jsonl")
    df.to_json(output_file, orient='records', lines=True)
    print("success")

if __name__ == "__main__":
    main()
