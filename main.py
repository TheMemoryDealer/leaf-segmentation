import matplotlib.pyplot as plt
import numpy as np
import csv, os, sys, glob, statistics, cv2, re
from termcolor import colored
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

IN_DIR = './data'

plt.rcParams["figure.figsize"] = (10,6) # fixed figure size

# GLOBAL SOTRAGE
img_raw_store = [] # store raw img here
img_label_store = [] # store labels here
name_store = [] # store img names
segmented_store = [] # store the answers here
results_store = [] # store segmentation applied
complete_store = [] # store leaf counts
kmeans_store = [] # store to show kmeans clustering segmentation
almost_final_store = [] # store almost final result
leaf_detected_store1 = [] # store num leaf detected (watershed)
leaf_detected_store2 = [] # store num leaf detected (k-means)
distance_store = [] # store distances before watershed
watershed_store = [] # store watershed results


def mother_of_plots(img_raw_store, img_label_store, segmented_store,
                    complete_store, kmeans_store, watershed_store):
    indx = 1
    for j in range(16): # through imgs
        ax1 = plt.subplot(16, 6, indx)
        plt.imshow(img_raw_store[j]) # raw
        plt.axis('off')
        indx += 1
        ax2 = plt.subplot(16, 6, indx)
        plt.imshow(img_label_store[j]) # label
        plt.axis('off')
        indx += 1
        ax3 = plt.subplot(16 ,6, indx)
        plt.imshow(segmented_store[j], cmap=plt.cm.gray) # thresh
        plt.axis('off')
        indx += 1
        ax4 = plt.subplot(16 ,6, indx)
        plt.imshow(kmeans_store[j]) # segm
        plt.axis('off')
        indx += 1
        ax5 = plt.subplot(16 ,6, indx)
        plt.imshow(watershed_store[j], cmap=plt.cm.nipy_spectral) # watershed
        plt.axis('off')
        indx += 1
        ax6 = plt.subplot(16, 6, indx)
        plt.imshow(complete_store[j]) # detected
        plt.axis('off')
        indx += 1

        if j == 0: # draw titles only on first row
            ax1.set_title("Input", fontweight='bold')
            ax2.set_title("Label", fontweight='bold')
            ax3.set_title("Thresh", fontweight='bold')
            ax4.set_title("K-means", fontweight='bold')
            ax5.set_title("Watershed", fontweight='bold')
            ax6.set_title("Detected", fontweight='bold')

    plt.show()      # <--------------------


def step_plot(img_raw_store, img_label_store, segmented_store,
              complete_store, kmeans_store, almost_final_store,
              leafs, leaf_detected_store2, name_store, results_store,
              leaf_detected_store1, distance_store, watershed_store):
    ds_store = [] # store ds score to later get mean
    leaf_error1 = [] # store detection error (watershed)
    leaf_error2 = [] # store detection error (k-means)
    name_store = list(set(name_store)) # remove duplicates
    name_store.sort() # sort names
    for i in range(len(img_raw_store)):
        # print(dice(segmented_store[i], img_label_store[i]))
        ax1 = plt.subplot(3, 3, 1)
        plt.imshow(img_raw_store[i])
        plt.axis('off')
        ax1.set_title("Input", fontweight='bold')

        ax2 = plt.subplot(3, 3, 2)
        plt.imshow(img_label_store[i])
        plt.axis('off')
        ax2.set_title('Label', fontweight='bold')

        ax3 = plt.subplot(3, 3, 3)
        plt.imshow(segmented_store[i], cmap=plt.cm.gray)
        plt.axis('off')
        ax3.set_title('Threshold', fontweight='bold')

        ax4 = plt.subplot(3, 3, 4)
        plt.imshow(results_store[i], cmap=plt.cm.gray)
        plt.axis('off')
        ax4.set_title('Bitwise', fontweight='bold', color="red")

        ax5 = plt.subplot(3, 3, 5)
        plt.imshow(distance_store[i], cmap=plt.cm.gray)
        plt.axis('off')
        ax5.set_title('Distances', fontweight='bold')

        ax6 = plt.subplot(3, 3, 6)
        plt.imshow(watershed_store[i], cmap=plt.cm.nipy_spectral)
        plt.axis('off')
        ax6.set_title('Watershed', fontweight='bold', color="blue")

        ax7 = plt.subplot(3, 3, 7)
        plt.imshow(kmeans_store[i])
        plt.axis('off')
        ax7.set_title('K-means', fontweight='bold')

        ax8 = plt.subplot(3, 3, 8)
        plt.imshow(almost_final_store[i], cmap=plt.cm.gray)
        plt.axis('off')
        ax8.set_title('Post-proc', fontweight='bold')

        ax9 = plt.subplot(3, 3, 9)
        plt.imshow(complete_store[i])
        plt.axis('off')
        ax9.set_title('Detected', color="purple", fontweight='bold')
        plt.figtext(0.15, 0.02, "DS - {}".format(dice(segmented_store[i], img_label_store[i])),
                    ha="center",
                    fontsize=16,
                    bbox={"facecolor":"red", "alpha":0.7, "pad":3})
        plt.figtext(0.85, 0.02, "Detected (K-Means) \n {}/{}".format(leaf_detected_store2[i],
                    leafs[i]),
                    ha="center",
                    fontsize=16,
                    bbox={"facecolor":"purple", "alpha":0.7, "pad":3})
        plt.figtext(0.5, 0.95, "{}.png".format(name_store[i]),
                    ha="center",
                    fontsize=16,
                    bbox={"facecolor":"yellow", "alpha":0.7, "pad":3})
        plt.figtext(0.5, 0.02, "Detected (Watershed) \n {}/{}".format(leaf_detected_store1[i],
					leafs[i]),
                    ha="center",
                    fontsize=16,
                    bbox={"facecolor":"blue", "alpha":0.7, "pad":3})
        plt.show()     # <--------------------
        ds_store.append(dice(segmented_store[i], img_label_store[i])) # add to later get mean
        leaf_error2.append(leafs[i] - leaf_detected_store2[i]) # add to later get mean
        leaf_error1.append(leafs[i] - leaf_detected_store1[i]) # same here
    print('DS mean - ', colored(statistics.mean(ds_store), 'red'))
    print('Leaf Detection difference mean - ',
		colored(statistics.mean(leaf_error1), 'cyan')) # watershed err
    print('Leaf Detection difference mean - ',
		colored(statistics.mean(leaf_error2), 'magenta')) # kmeans err

def dice(mask, gt):
    mask = np.asarray(mask).astype(np.bool)
    gt = np.asarray(gt).astype(np.bool)
    gt = gt[:,:,1] # gt comes in all colour spaces. Pick one
    # print(mask.shape)     # <------ Debug
    # print(gt.shape)
    if mask.shape != gt.shape:
        raise ValueError("Shape mismatch: mask and gt must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(mask, gt) # where both intersect
    return round(2. * intersection.sum() / (mask.sum() + gt.sum()),2)

def kmeans(img, kmeans_store):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    kmeans_store.append(segmented_image)
    # segmented_image = cv2.GaussianBlur(segmented_image,(3,3),0) # play with this
    # plt.imshow(segmented_image) # <------ DEBUG
    # plt.show() # <------ DEBUG
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(segmented_image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(img.shape)
    return masked_image

def wshed(m):
	distance = ndi.distance_transform_edt(m) # calc distances
	distance_store.append(-distance)
	# get coordinates from distances. play around with footprint
	coords = peak_local_max(distance, footprint=np.ones((12, 12)), labels=m)
	# prep mask array
	maskk = np.zeros(distance.shape, dtype=bool)
	# check if mask aligns with coordinates
	maskk[tuple(coords.T)] = True
	# label the mask
	markers, _ = ndi.label(maskk)
	# do segmentation
	labels = watershed(-distance, markers, mask=m)
	return labels

def main():
    print('Input directory: {}'.format(IN_DIR))
    img_paths = glob.glob(os.path.join(IN_DIR, '*.png'))
    img_paths.sort()
    print(colored('{} image paths loaded'.format(len(img_paths)), 'red'))
    # sort images
    img_raw_dir = [ x for x in img_paths if "rgb" in x ]
    img_label_dir = [ x for x in img_paths if "label" in x ]
    print(colored("\n".join(img_raw_dir), 'green'))
    print(colored("\n".join(img_label_dir), 'yellow'))
    df = pd.read_csv('./Leaf_counts.csv', names=['name','leafs']) # read real leafs
    leafs = df["leafs"].tolist() # convert to list
    print('Actual leaf count - ', leafs)
    # read in raw images and store stuff to mem
    leafi = 0
    for img in img_paths:
        # get img names here
        img_name = img
        m = re.search('a/(.+?)_', img_name)
        if m: # grab names
            name = m.group(1)
            name_store.append(name) # add them to list
        if 'rgb' in img: # this img is to be processed
            imgg = cv2.imread(img) # all imgs on OpenCV are BGR by default.
            img = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
            img_raw_store.append(img)
            # now do the thresholding
            imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            low = (33,78,67)
            high = (151,255,255)
            mask = cv2.inRange(imgHSV, low, high)
            # play around with trans
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 1)))
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2)))
            mask = cv2.erode(mask,np.ones((1, 1)),iterations = 2)
            segmented_store.append(mask)
            labels = wshed(mask)
            watershed_store.append(labels)
            # measure properties of labeled image regions
            regions = regionprops(labels)
            regions = [r for r in regions if r.area > 60] # sanity check
            print(colored('Leaf detected (from Watershed) : ', 'cyan'),
                  len(regions), '/', leafs[leafi])
            leaf_detected_store1.append(len(regions))

            # get threshold results
            result = cv2.bitwise_and(imgHSV, imgHSV, mask=mask)
            result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
            results_store.append(result)

            # <------ DEBUG SECTION
            # plt.imshow(result)
            # plt.axis('off')
            # plt.show()

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = result.copy() # apparently normal assignment just wont work
            masked_image = kmeans(img, kmeans_store) # apply kmeans segmentation
            masked_image = cv2.blur(masked_image, (6, 6)) # blur a bit
            gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            gray_blurred = cv2.blur(gray, (6, 6)) # blur a bit
            gray_blurred = cv2.erode(gray_blurred,np.ones((1, 1)),iterations = 1)
            # gray_blurred = cv2.morphologyEx(gray_blurred, cv2.MORPH_CLOSE, np.ones((5, 5)))
            almost_final_store.append(gray_blurred)
            # apply Hough transform on the blurred image.
            detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 1, 10, param1 = 50,
                        param2 = 18, minRadius = 0, maxRadius = 44)
            # araw circles that are detected.
            if detected_circles is not None:
                # convert the circle parameters a, b and r to integers.
                detected_circles = np.uint16(np.around(detected_circles))
                counter = 1
                print(colored('Leaf detected (from K-means): ', 'magenta'),
                      len(detected_circles[0, :]), '/', leafs[leafi])
                leaf_detected_store2.append(len(detected_circles[0, :]))
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    # draw the circumference of the circle.
                    cv2.circle(img, (a, b), r, (255, 0, 0), 2)
                    # draw a small circle (of radius 1) to show the center.
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                    cv2.putText(img, "{}".format(counter), (a, b - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 50, 200), 2)
                    counter+=1
                leafi+=1
            else: # in case no circles are detected
                leaf_detected_store2.append(0)
                print('Leaf detected : ', '0', '/', leafs[leafi])
                leafi+=1
            complete_store.append(img)

            ax1 = plt.subplot(1, 2, 2)
            plt.imshow(img)
            ax1.set_title("Detected", color="purple")
            plt.axis('off')
            ax2 = plt.subplot(1, 2, 1)
            ax2.set_title("Watershed", color="blue")
            plt.imshow(labels, cmap=plt.cm.nipy_spectral)
            plt.axis('off')
            plt.show()

        elif 'label' in img: # this is label image
            img = cv2.imread(img)
            # px = np.count_nonzero(img) # <------ DEBUG
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_label_store.append(img)
            # print('px - ', px) # <------ DEBUG
        else:
            print('THIS SHOULD NOT BE HERE')
        print('-' * 10)
    step_plot(img_raw_store, img_label_store, segmented_store,
              complete_store, kmeans_store, almost_final_store,
              leafs, leaf_detected_store2, name_store, results_store,
              leaf_detected_store1, distance_store, watershed_store)

    mother_of_plots(img_raw_store, img_label_store, segmented_store,
                    complete_store, kmeans_store, watershed_store)

if __name__ == "__main__":
   main() # :)