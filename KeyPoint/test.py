import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def detect_features(image, algorithm):
    if algorithm == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04)
    elif algorithm == 'ORB':
        detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    elif algorithm == 'AKAZE':
        detector = cv2.AKAZE_create(threshold=0.001)
    elif algorithm == 'BRISK':
        detector = cv2.BRISK_create(thresh=30, octaves=3)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, algorithm):
    if descriptors1 is None or descriptors2 is None:
        return []

    if algorithm in ['SIFT', 'AKAZE', 'BRISK']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 75%의 매치만 사용
    good_matches = matches[:int(len(matches) * 0.75)]

    return good_matches

def preprocess_image(image):
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 샤프닝
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def visualize_features_and_matches(image1, image2, keypoints1, keypoints2, matches, algorithm):
    # 특징점 시각화
    img1_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0))
    img2_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0))

    # 매칭 결과 시각화
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 결과 출력
    plt.figure(figsize=(20, 10))
    plt.subplot(221), plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'{algorithm} Keypoints - Image 1'), plt.axis('off')
    plt.subplot(222), plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'{algorithm} Keypoints - Image 2'), plt.axis('off')
    plt.subplot(212), plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'{algorithm} Matches'), plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_image_pair(image1_path, image2_path, algorithm):
    # 이미지 읽기
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print(f"Warning: Unable to read image {image1_path} or {image2_path}")
        return

    # 이미지 전처리
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # 특징점 검출
    keypoints1, descriptors1 = detect_features(image1, algorithm)
    keypoints2, descriptors2 = detect_features(image2, algorithm)

    # 특징점 매칭
    matches = match_features(descriptors1, descriptors2, algorithm)

    # 결과 시각화
    visualize_features_and_matches(image1, image2, keypoints1, keypoints2, matches, algorithm)

    return len(keypoints1), len(keypoints2), len(matches)

def analyze_and_plot(results):
    algorithms = list(results.keys())
    keypoints1 = [results[alg]['keypoints1'] for alg in algorithms]
    keypoints2 = [results[alg]['keypoints2'] for alg in algorithms]
    matches = [results[alg]['matches'] for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, keypoints1, width, label='Keypoints Image 1')
    rects2 = ax.bar(x, keypoints2, width, label='Keypoints Image 2')
    rects3 = ax.bar(x + width, matches, width, label='Matches')

    ax.set_ylabel('Count')
    ax.set_title('Comparison of Feature Detection Algorithms')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()

    plt.tight_layout()
    plt.show()

# 메인 실행 부분
image_folder = "./dataset_split/test/damaged"
algorithms = ['SIFT', 'ORB', 'AKAZE', 'BRISK']

# 첫 두 개의 이미지에 대해 모든 알고리즘 실행
image_files = sorted(os.listdir(image_folder))
image1_path = os.path.join(image_folder, image_files[0])
image2_path = os.path.join(image_folder, image_files[1])

results = {}

for algorithm in algorithms:
    print(f"\nProcessing {algorithm}:")
    num_keypoints1, num_keypoints2, num_matches = process_image_pair(image1_path, image2_path, algorithm)
    print(f"Number of Keypoints in Image 1: {num_keypoints1}")
    print(f"Number of Keypoints in Image 2: {num_keypoints2}")
    print(f"Number of Matches: {num_matches}")
    
    results[algorithm] = {
        'keypoints1': num_keypoints1,
        'keypoints2': num_keypoints2,
        'matches': num_matches
    }

# 결과 분석 및 그래프 생성
analyze_and_plot(results)