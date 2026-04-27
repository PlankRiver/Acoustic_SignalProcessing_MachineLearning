import os
import datetime

#定义一个目录名称
project_folder = 'my_project'
data_folder = os.path.join(project_folder, 'data')
output_folder = os.path.join(project_folder, 'output')
print(data_folder,output_folder,sep='\n')

#创建目录结构
print(f'创建目录:{data_folder}')
os.makedirs(data_folder, exist_ok=True)
print(f'创建目录:{output_folder}')
os.makedirs(output_folder, exist_ok=True)

#在data目录下创建文件
file_path = os.path.join(data_folder, 'raw_data.txt')
with open(file_path,'w',encoding='utf-8') as f:
    f.write('这是一个python写的os操作系统交互文件')

if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f'<UNK>{file_size}<UNK>')