import os

file_path = '/home/Datasets/DOTA/Annotations/'
file_list = os.listdir(file_path)

if __name__ == '__main__':
    with open('/home/Datasets/DOTA/ImageSets/test.txt', 'w') as f:
        for file_name in file_list:
            f.write(file_name[:-4] + '\n')
