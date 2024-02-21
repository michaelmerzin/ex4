import cv2 as cv
from matplotlib import pyplot as plt

import ex3
import numpy as np


# In this exercise, you will need to blend between a low-resolution image and a high-resolution part of it
# using the material taught in class up until now. In the Moodel, under ”Exercise 4” -> ”Exercise Inputs”
# (found here) you are given 2 low-resolution images and 2 high-resolution parts of the image. Your task
# is to design an algorithm that takes such a pair and blends them with each other.\


# we will use sift to find the matching points between the low resolution image and the high resolution image
# we will use the matching points to find the homography matrix
# we will use the homography matrix to warp the high resolution image to the low resolution image
# we will use the warped image to blend the low resolution image and the high resolution image

def find_homography_matrix(img1, img2):
    # find the keypoints and descriptors with SIFT
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # create a BFMatcher object
    bf = cv.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # find the matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # find the homography matrix
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return M


def warp_image(img1, img2, homography_matrix):
    # warp the image
    width = img1.shape[1]
    height = img1.shape[0]
    # tranpose the homography matrix
    homography_matrix = np.linalg.inv(homography_matrix)
    warped_img = cv.warpPerspective(img2, homography_matrix, (width, height))
    # convert to bgr to rgb
    warped_img_helper = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)
    plt.imshow(warped_img_helper)
    plt.show()
    return warped_img


def make_mask_for_high_resolution_image_after_homography_over_columns(high_res_img):
    # go over each column and find the first non zero pixel and the last non zero pixel ,then between them make the pixels 1
    mask = np.zeros(high_res_img.shape[:2], dtype=np.uint8)
    for i in range(high_res_img.shape[1]):
        start = -1
        end = -1
        for j in range(high_res_img.shape[0]):
            if high_res_img[j, i, 0] != 0 or high_res_img[j, i, 1] != 0 or high_res_img[j, i, 2] != 0:
                mask[j, i] = 1
                start = j
                break
        for j in range(high_res_img.shape[0] - 1, -1, -1):
            if high_res_img[j, i, 0] != 0 or high_res_img[j, i, 1] != 0 or high_res_img[j, i, 2] != 0:
                mask[j, i] = 1
                end = j
                break
        for j in range(high_res_img.shape[0]):
            if start <= j <= end:
                mask[j, i] = 1

    return mask

#def make_mask_for_high_resolution_image_after_homography_by_png(high_res_img):

def make_mask_for_high_resolution_image_after_homography_over_rows(high_res_img):
    # go over each row and find the first non zero pixel and the last non zero pixel ,then between them make the pixels 1
    mask = np.zeros(high_res_img.shape[:2], dtype=np.uint8)
    for i in range(high_res_img.shape[0]):
        start = -1
        end = -1
        for j in range(high_res_img.shape[1]):
            if high_res_img[i, j, 0] != 0 or high_res_img[i, j, 1] != 0 or high_res_img[i, j, 2] != 0:
                mask[i, j] = 1
                start = j
                break
        for j in range(high_res_img.shape[1] - 1, -1, -1):
            if high_res_img[i, j, 0] != 0 or high_res_img[i, j, 1] != 0 or high_res_img[i, j, 2] != 0:
                mask[i, j] = 1
                end = j
                break
        for j in range(high_res_img.shape[1]):
            if start <= j <= end:
                mask[i, j] = 1

    return mask


def blend_with_mask(img1, img2, mask):
    # blurr the mask
    mask = cv.GaussianBlur(mask, (5, 5), 0)
    blended_img = img1 * (1 - mask) + img2 * mask
    return blended_img


def blending_low_high_resolution_for_dessert(low_res_img, high_res_img):
    # find the homography matrix
    # plot low res image
    plt.imshow(low_res_img)
    plt.show()

    homography_matrix = find_homography_matrix(low_res_img, high_res_img)
    # warp the image
    warped_img = warp_image(low_res_img, high_res_img, homography_matrix)
    # blend the images
    # convert to rgb from bgr low_res_img = cv.cvtColor(low_res_img, cv.COLOR_BGR2RGB)
    # low_res_img = cv.cvtColor(low_res_img, cv.COLOR_BGR2RGB)
    # warped_img = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)
    plt.imshow(low_res_img)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    mask = make_mask_for_high_resolution_image_after_homography_over_rows(warped_img)

    # show the mask
    plt.imshow(mask, cmap='gray')
    plt.show()
    # multiply the mask to be 3 channels
    mask = np.stack([mask, mask, mask], axis=2)
    blended_img = blend_with_mask(low_res_img, warped_img, mask)

    return blended_img


def blending_low_high_resolution_for_lake(low_res_img, high_res_img):
    # find the homography matrix
    # plot low res image
    plt.imshow(low_res_img)
    plt.show()

    homography_matrix = find_homography_matrix(low_res_img, high_res_img)
    # warp the image
    warped_img = warp_image(low_res_img, high_res_img, homography_matrix)
    # blend the images
    # convert to rgb from bgr low_res_img = cv.cvtColor(low_res_img, cv.COLOR_BGR2RGB)
    # low_res_img = cv.cvtColor(low_res_img, cv.COLOR_BGR2RGB)
    # warped_img = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)
    plt.imshow(low_res_img)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    mask = make_mask_for_high_resolution_image_after_homography_over_columns(warped_img)
    # show the mask
    plt.imshow(mask, cmap='gray')
    plt.show()
    # multiply the mask to be 3 channels
    mask = np.stack([mask, mask, mask], axis=2)
    blended_img = blend_with_mask(low_res_img, warped_img, mask)

    return blended_img
