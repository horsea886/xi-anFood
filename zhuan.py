import json

def dat_to_json(input_file, output_file):
    result_dict = {}

    with open(input_file, 'r') as dat_file:
        for line in dat_file:
            # 分割每行数据（使用双冒号分隔符）
            parts = line.strip().split('::')

            # 验证数据格式（确保每行有4列）
            if len(parts) == 4:
                user_id = parts[0]
                # 构建指定格式的嵌套字典
                result_dict[user_id] = {
                    "gender": parts[1],
                    "age": int(parts[2]),
                    "occupation": int(parts[3])
                }

    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, indent=4, ensure_ascii=False)


# 示例调用
dat_to_json('data/ml-1m/users.dat', 'data/users.json')