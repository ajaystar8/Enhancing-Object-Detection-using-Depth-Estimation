import argparse
import os
from glob import glob

import numpy as np

import config

EPS = 0.01
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416


def IOU(bbox, centroids):
    similarities = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = bbox
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)
    return np.array(similarities)


def avg_IOU(bbox_dimensions, centroids):
    total_bboxes = bbox_dimensions.shape[0]
    sum_of_iou = 0.
    for i in range(total_bboxes):
        sum_of_iou += max(IOU(bbox_dimensions[i], centroids))
    return sum_of_iou / total_bboxes


def write_anchors_to_file(centroids, anchor_file: str):
    anchors = centroids.copy()
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_idx = np.argsort(areas)
    sorted_anchors = np.array(anchors[sorted_idx]).reshape(-1, 2)
    np.savetxt(anchor_file, sorted_anchors, fmt="%.2f")


def perform_kmeans(bbox_dimensions: np.ndarray, centroids: np.ndarray, anchor_file: str):
    total_bboxes = bbox_dimensions.shape[0]
    num_clusters, dim = centroids.shape

    iteration = 0
    prev_centroids = np.ones(total_bboxes) * -1

    while True:
        iteration += 1
        print(f'Iteration {iteration}')
        curr_distances = []

        for i in range(total_bboxes):
            bbox_to_centroid_distances = 1 - IOU(bbox_dimensions[i], centroids)
            curr_distances.append(bbox_to_centroid_distances)
        curr_distances = np.array(curr_distances)

        assigned_centroids = np.argmin(curr_distances, axis=1)

        # terminating condition
        if (assigned_centroids == prev_centroids).all():
            print("K-Means Converged!")
            print(f"Centroids: {centroids}")
            write_anchors_to_file(centroids, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((num_clusters, dim), dtype=np.float32)
        for i in range(total_bboxes):
            centroid_sums[assigned_centroids[i]] += bbox_dimensions[i]
        for j in range(num_clusters):
            centroids[j] = centroid_sums[j] / np.sum(assigned_centroids == j)

        # update variables
        prev_centroids = assigned_centroids.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-label_dir',
                        default="/Users/ajay/Documents/NEU_Academics/Sem-1/Computer_Vision/Group_Project/NYUD/labels",
                        help="Directory with annotations in YOLO format")
    parser.add_argument('-output_dir',
                        default="/Users/ajay/Documents/NEU_Academics/Sem-1/Computer_Vision/Group_Project/NYUD/anchors",
                        help="Directory where the generated anchors will be saved")
    parser.add_argument('-num_clusters', default=9, help="Number of clusters\n", type=int)
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load the heights and widths of all the bounding boxes
    labels_file_paths = glob(args.label_dir + "/*.txt")
    bbox_dimensions = []
    for path in labels_file_paths:
        with open(path, "r") as f:
            for line in f.readlines():
                class_id, x, y, w, h = line.rstrip().split(" ")
                bbox_dimensions.append((float(w), float(h)))
    bbox_dimensions = np.array(bbox_dimensions)
    total_bboxes = bbox_dimensions.shape[0]

    # Generate anchors
    anchor_file = os.path.join(args.output_dir, f"anchors_{args.num_clusters}.txt")

    centroids = []
    for bbox in config.ANCHORS:
        centroids.extend(bbox)
    centroids = np.array(centroids)

    # perform kmeans
    perform_kmeans(bbox_dimensions, centroids, anchor_file)


if __name__ == "__main__":
    main()
