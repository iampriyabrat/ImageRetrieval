import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Find the index of the closest central point to the each sift descriptor. 
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.  
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
        else:
            dist = distance.euclidean(image, center[i]) 
            if(dist < count):
                ind = i
                count = dist
    return ind

def images_print(images, feature_vectors, test_vectors, test):
    loop_num = 0
    samples_name = ["pizza", "stop_sign", "sunflower"]
    for i in samples_name:
        cv2.imwrite(str(i)+".jpg", test[i][4])
        closest_images = closests(feature_vectors, test_vectors[i][4])
        x = []
        for ind in range(len(closest_images)):
            x.append(cv2.resize(images[closest_images[ind][0]][closest_images[ind][1]],(250,250))) 
        img_concatanete = np.concatenate((x[0],x[1],x[2],x[3],x[4]),axis=1)
        cv2.imwrite('the_closest_images_to_'+ str(i)+".jpg",img_concatanete)


def closests(images, test):
    img = [["", 0], ["", 0], ["",0], ["",0], ["",0]]
    dist = [np.inf, np.inf, np.inf, np.inf, np.inf]
    
    for key, value in images.items():
        for ind in range(len(value)):
            dist_val = distance.euclidean(test, value[ind])
            for i in range(len(dist)):
                if(dist_val < dist[i]):
                    dist[i] = dist_val
                    img[i][0] = key
                    img[i][1] = ind
                    break
    return img

# Creates descriptors using sift 
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def sift_features(images):
    print("Extracting Features...")
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img,None)
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]


# A k-means clustering algorithm who takes 2 parameter which is number 
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeans(k, descriptor_list):
    print("kmeans")
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
    

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    print("Train bovw")
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature
    

# 1-NN algorithm. We use this for predict the class of test images.
# Takes 2 parameters. images is the feature vectors of train images and tests is the feature vectors of test images
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
def knn(images, tests):
    print("Classifaction")
    num_test = 0
    correct_predict = 0
    class_based = {}
    
    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0] # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a" #predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key
            
            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]
    


# Calculates the average accuracy and class based accuracies.  
def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))
         
# return a dictionary that holds all images category by category. 
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

def main():
    print("Loading images to train...")
    images = load_images_from_folder('dataset/train')  # take all images category by category 
    print("Loading images to test...")
    test = load_images_from_folder('dataset/query') # take test images
    sifts = sift_features(images) 
    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0] 
    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[1] 
    # Takes the central points which is visual words    
    visual_words = kmeans(100, descriptor_list) 
    print("kmeans over")
    # Creates histograms for train data    
    bovw_train = image_class(all_bovw_feature, visual_words) 
    # Takes the sift features that is seperated class by class for test data
    test_bovw_feature = sift_features(test)[1]
    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words) 

    # name_dict = {}
    # label_count = 0 
    # train_labels = np.array([])
    # for key,value in images.items():
    #     name_dict[str(label_count)] = key
    #     for im in value:
    #         train_labels = np.append(train_labels, label_count)
    #     label_count += 1
    # scale = StandardScaler().fit(bovw_train)
    # bovw_train = scale.transform(bovw_train)
    # error=[]
    # #knn for k=1 to 5
    # for i in range(1, 5):
    #     classifier = KNeighborsClassifier(n_neighbors=i)
    #     classifier.fit(bovw_train, train_labels)
    #     for test_key, test_val in tests.items():
    #         test_val = scale.transform(test_val)
    #         lb=classifier.predict(test_val)
    #         error.append(np.mean(lb != test_key))
    # #error rate
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 5), error, color='red', linestyle='dashed', marker='o',
    #      markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')


    # Call the knn function    
    results_bowl = knn(bovw_train, bovw_test)
    # Calculates the accuracies and write the results to the console. 
    print(results_bowl)    
    accuracy(results_bowl) 
    images_print(images, bovw_train, bovw_test, test)

if __name__=="__main__":
    main()