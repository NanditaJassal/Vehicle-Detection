import os
import pickle
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from config import root_data_non_vehicle, root_data_vehicle, feat_extraction_params
from functions_detection import draw_boxes
from functions_detection import search_windows
from functions_detection import slide_window
from functions_feat_extraction import extract_features_from_file_list
from project_5_utils import get_file_list_recursively


if __name__ == '__main__':

    
    cars = get_file_list_recursively(root_data_vehicle)
    cars_not = get_file_list_recursively(root_data_non_vehicle)

    print('Now extract car features :-')
    extract_car_features = extract_features_from_file_list(cars, feat_extraction_params)

    print('Now Extracting non-car features :-')
    extract_cars_not_features = extract_features_from_file_list(notcars, feat_extraction_params)

    X = np.vstack((extract_car_features, extract_cars_not_features)).astype(np.float64)

    
    feature_scaler = StandardScaler().fit(X)  
    scaled_X = feature_scaler.transform(X)

    
    y = np.hstack((np.ones(len(extract_car_features)), np.zeros(len(extract_cars_not_features))))

    
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('The feature vector length is equal to', len(X_train[0]))

    
    svc = LinearSVC()  

    
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'seconds remaining')

    
    print('Test Accuracy = ', round(svc.score(X_test, y_test), 4))

    
    with open('data/svm_trained.pickle', 'wb') as f:
        pickle.dump(svc, f)
    with open('data/feature_scaler.pickle', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open('data/feat_extraction_params.pickle', 'wb') as f:
        pickle.dump(feat_extraction_params, f)

    
    test_img_dir = 'test_images'
    for test_img in os.listdir(test_img_dir):
        image = cv2.imread(os.path.join(test_img_dir, test_img))

        h, w, c = image.shape
        draw_image = np.copy(image)

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[h//2, None],
                               xy_window=(64, 64), xy_overlap=(0.8, 0.8))

        hot_windows = search_windows(image, windows, svc, feature_scaler, feat_extraction_params)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
        plt.show()

