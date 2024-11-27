import pandas as pd
import os
import csv

def create_csv():
    '''
    이미지 경로와 target 매칭해 csv(Image Path, Label) 파일 생성

    Image Path = (../Data/augmentation 또는 ../Data/) + image file name
    Label = control: 0 / 노균병:1 / 노균병 유사 :2 / 흰가루병: 3 / 흰가루병 유사: 4

    :Input:
    Image Path = ../Data/augmentation 또는 ../Data/
    :Output:
    csv
    '''


    # CSV 파일 경로와 열 제목
    csv_file = r'/Output/image_data.csv'
    csv_columns = ['Image Path', 'Label']

    base_path = r"C:\data\Training\01.원천데이터"

    # 폴더 경로
    folder_paths = ['control', 'downy', 'not_downy', 'not_powdery', 'powdery']


    # CSV 파일 생성 및 열 제목 작성
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

        # 폴더 탐색 및 이미지 정보 작성
        for folder_path in folder_paths:
            path = os.path.join(base_path, folder_path)
            image_files = os.listdir(path)
            print(path)
            for image_file in image_files:

                # image_path = os.path.join(path, image_file)
                image_path = path + "/" + image_file

                # 진딧물
                if folder_path == "downy":
                    label = 1
                # 담배가루이-선충
                elif folder_path == "not_downy":
                    label = 2
                # 담배가루이-약충
                elif folder_path == "not_powdery":
                    label = 3
                # control(토마토잎, 오이잎)
                elif folder_path == "powdery":
                    label = 4
                else:
                    label = 0

                # CSV 파일에 이미지 정보 작성
                writer.writerow({'Image Path': image_path, 'Label': label})
                print("완료되었습니다.")

def main():


    create_csv()



if __name__ == '__main__':
    main()