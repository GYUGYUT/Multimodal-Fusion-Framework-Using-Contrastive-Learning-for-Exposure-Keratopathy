import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_processor, transform=None, num_classes=4, ext='.jpg', broad_mode=False, broad_csv_file=None):
        """
        Args:
            csv_file (str): CSV/Excel 파일 경로
            image_folder (str): 이미지 폴더 경로
            image_processor: Hugging Face AutoImageProcessor 인스턴스
            num_classes (int): 전체 클래스 수 (원-핫 인코딩 시 사용)
            ext (str): 이미지 확장자 (예: '.png')
            broad_mode (bool): broad 기준 grade별 개수 맞추기 활성화 여부
            broad_csv_file (str): broad 엑셀 파일 경로 (broad_mode=True일 때 필요)
        """
        # 엑셀/CSV 자동 판별
        if csv_file.endswith('.csv'):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = pd.read_excel(csv_file)
        # broad 기준 grade 시퀀스와 등장 순서에 맞춰 정렬
        if broad_mode and broad_csv_file is not None:
            if broad_csv_file.endswith('.csv'):
                broad_data = pd.read_csv(broad_csv_file)
            else:
                broad_data = pd.read_excel(broad_csv_file)
            broad_grade_seq = list(broad_data['grade'])
            grade_to_indices = {g: self.data[self.data['grade'] == g].index.tolist() for g in self.data['grade'].unique()}
            ptr = {g: 0 for g in grade_to_indices}
            new_indices = []
            for g in broad_grade_seq:
                indices = grade_to_indices.get(g, [])
                if not indices:
                    new_indices.append(self.data.index[0])
                else:
                    idx = indices[ptr[g] % len(indices)]
                    new_indices.append(idx)
                    ptr[g] += 1
            self.data = self.data.loc[new_indices].reset_index(drop=True)
        else:
            # grade 기준 정렬(기존 동작)
            self.data = self.data.sort_values(by='grade').reset_index(drop=True)
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.transform = transform
        self.image_folder = image_folder
        self.ext = ext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # SMC 데이터셋 기준: 'Detailed ID', 'grade' 컬럼 사용
        img_id = str(row['Detailed ID'])
        raw_label = int(row['grade'])

        img_path = os.path.join(self.image_folder, img_id + self.ext)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        processed = self.image_processor(image, return_tensors="pt")
        image = processed["pixel_values"].squeeze(0)
        # 라벨을 텐서로 변환 (dtype: long)
        label = torch.tensor(raw_label, dtype=torch.long)
        return image, label 