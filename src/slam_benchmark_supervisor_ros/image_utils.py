#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import traceback
from os import path
from PIL.ImageDraw import floodfill
from PIL import Image

import rospy
import cv2
import numpy as np
import copy
import yaml

from performance_modelling_py.utils import print_info, backup_file_if_exists, print_error, print_fatal


def mse(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(image_a, image_b):
    # compute the mean squared error and structural similarity
    # index for the images
    image_a = cv2.resize(image_a, (0, 0), fx=0.25, fy=0.25)
    image_b = cv2.resize(image_b, (0, 0), fx=0.25, fy=0.25)
    m = mse(image_a, image_b)
    return m


def map_changed(previous_map_file_path, current_map_file_path, size_change_threshold, map_change_threshold):
    previous_map_image = cv2.imread(previous_map_file_path)
    current_map_image = cv2.imread(current_map_file_path)

    size_diff = float(current_map_image.shape[0] * current_map_image.shape[1] - previous_map_image.shape[0] * previous_map_image.shape[1]) / (
            current_map_image.shape[0] * current_map_image.shape[1])

    if size_diff * 100 > size_change_threshold:
        # area changed more than threshold
        return True
    elif size_diff != 0:  # increasing map size
        # TODO get rid of this code (from predictive_benchmarking)
        # note: second image could be slightly smaller than the first due to small corrections
        # we crop the first image to ensure it's always smaller than the second
        min_width = previous_map_image.shape[1] if previous_map_image.shape[1] < current_map_image.shape[1] else current_map_image.shape[1]
        min_height = previous_map_image.shape[0] if previous_map_image.shape[0] < current_map_image.shape[0] else current_map_image.shape[0]
        previous_map_image = previous_map_image[0:min_height, 0:min_width]

        # find contained image
        result = cv2.matchTemplate(current_map_image, previous_map_image, cv2.TM_SQDIFF_NORMED)
        _, _, min_loc, _ = cv2.minMaxLoc(result)
        min_loc_x, min_loc_y = min_loc
        empty = np.full(shape=(current_map_image.shape[0], current_map_image.shape[1], 3), fill_value=205, dtype=np.uint8)
        empty[min_loc_y:min_loc_y + min_height, min_loc_x:min_loc_x + min_width] = previous_map_image

        # copy
        previous_map_image = empty

        # fit in bigger image
        empty = np.full(shape=(4000, 4000, 3), fill_value=205, dtype=np.uint8)
        empty[0:previous_map_image.shape[0], 0:previous_map_image.shape[1]] = previous_map_image
        previous_map_image = empty

        empty = np.full(shape=(4000, 4000, 3), fill_value=205, dtype=np.uint8)
        empty[0:current_map_image.shape[0], 0:current_map_image.shape[1]] = current_map_image
        current_map_image = empty

    try:
        diff = compare_images(current_map_image, previous_map_image)
        return diff > map_change_threshold
    except ValueError:
        rospy.logerr('save_map_snapshot: compare images error')
        return True


def save_map_image(original_map_msg, image_file_path, info_file_path, map_free_threshold, map_occupied_threshold):
    map_msg = copy.deepcopy(original_map_msg)
    map_image = np.zeros((map_msg.info.height, map_msg.info.width), dtype=np.uint8)

    for y in range(map_msg.info.height):
        for x in range(map_msg.info.width):
            i = x + (map_msg.info.height - y - 1) * map_msg.info.width
            if 0 <= map_msg.data[i] <= map_free_threshold:  # [0, free)
                map_image[y, x] = 254
            elif map_msg.data[i] >= map_occupied_threshold:  # (occ, 255]
                map_image[y, x] = 0
            else:  # [free, occ]
                map_image[y, x] = 205

    # save map image
    cv2.imwrite(image_file_path, map_image)

    # save map info
    with open(info_file_path, 'w') as yaml_file:
        yaml_dict = {
            'header': {
                'seq': map_msg.header.seq,
                'stamp': map_msg.header.stamp.to_sec(),
                'frame_id': map_msg.header.frame_id,
                },
            'info': {
                'map_load_time': map_msg.info.map_load_time.to_sec(),
                'resolution': map_msg.info.resolution,
                'width': map_msg.info.width,
                'height': map_msg.info.height,
                'origin': {
                    'position': {'x': map_msg.info.origin.position.x, 'y': map_msg.info.origin.position.y, 'z': map_msg.info.origin.position.z},
                    'orientation': {'x': map_msg.info.origin.orientation.x, 'y': map_msg.info.origin.orientation.y, 'z': map_msg.info.origin.orientation.z, 'w': map_msg.info.origin.orientation.w},
                },
            },
        }
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False)


def xiao_line(x0, y0, x1, y1):

    x = []
    y = []
    dx = x1 - x0
    dy = y1 - y0
    steep = abs(dx) < abs(dy)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dy, dx = dx, dy

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    gradient = float(dy) / float(dx)  # slope

    """ handle first endpoint """
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xpxl0 = int(xend)
    ypxl0 = int(yend)
    x.append(xpxl0)
    y.append(ypxl0)
    x.append(xpxl0)
    y.append(ypxl0 + 1)
    intery = yend + gradient

    """ handles the second point """
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xpxl1 = int(xend)
    ypxl1 = int(yend)
    x.append(xpxl1)
    y.append(ypxl1)
    x.append(xpxl1)
    y.append(ypxl1 + 1)

    """ main loop """
    for px in range(xpxl0 + 1, xpxl1):
        x.append(px)
        y.append(int(intery))
        x.append(px)
        y.append(int(intery) + 1)
        intery = intery + gradient

    if steep:
        y, x = x, y

    coords = zip(x, y)

    return coords


def color_diff(a, b):
    return np.sum(np.array(b) - np.array(a)) / len(a)


def color_abs_diff(a, b):
    return np.abs(color_diff(a, b))


def compute_ground_truth_from_stage_map(stage_world_folder, do_not_recompute=False, backup_if_exists=False):
    stage_map_file_path = path.join(stage_world_folder, "map.png")
    stage_map_info_file_path = path.join(stage_world_folder, "stage_world_info.yaml")
    ground_truth_map_file_path = path.join(stage_world_folder, 'map_ground_truth.pgm')
    if path.exists(ground_truth_map_file_path):
        print_info("file already exists: {}".format(ground_truth_map_file_path))
        if do_not_recompute:
            print_info("do_not_recompute: will not recompute the output image")
            return

    gt = Image.open(stage_map_file_path)
    print_info("opened image, mode: {mode}, image: {im}".format(mode=gt.mode, im=stage_map_file_path))

    if gt.mode != 'RGB':
        print('image mode is {mode} ({size}×{ch_num}), converting to RGB'.format(mode=gt.mode, size=gt.size, ch_num=len(gt.split())))
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", gt.size, (254, 254, 254))
        background.paste(gt)
        gt = background

    pixels = gt.load()

    # crop borders not containing black pixels (stage ignores non-black borders while placing the pixels in the simulated map, so they need to be ignored in the following calculations)
    w, h = gt.size
    top_border = 0
    for y in range(h):
        found = False
        for x in range(w):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                top_border = y
                found = True
                break
        if found:
            break
    bottom_border = h
    for y in range(h)[::-1]:
        found = False
        for x in range(w):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                bottom_border = y + 1
                found = True
                break
        if found:
            break
    left_border = 0
    for x in range(w):
        found = False
        for y in range(h):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                left_border = x
                found = True
                break
        if found:
            break
    right_border = w
    for x in range(w)[::-1]:
        found = False
        for y in range(h):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                right_border = x + 1
                found = True
                break
        if found:
            break

    print("crop box: left_border: {}, top_border: {}, right_border: {}, bottom_border: {}".format(left_border, top_border, right_border, bottom_border))
    gt = gt.crop(box=(left_border, top_border, right_border, bottom_border))
    pixels = gt.load()
    print('cropped image: {mode} ({size}×{ch_num})'.format(mode=gt.mode, size=gt.size, ch_num=len(gt.split())))

    # convert all free pixels to unknown pixels
    for i in range(gt.size[0]):
        for j in range(gt.size[1]):
            if color_abs_diff(pixels[i, j], (0, 0, 0)) < 150:  # black -> wall. color_abs_diff must be less than 205 - 0 (difference between occupied black and unknown grey)
                pixels[i, j] = (0, 0, 0)
            else:  # not black -> unknown space
                pixels[i, j] = (205, 205, 205)

    with open(stage_map_info_file_path, 'r') as info_file:
        info_yaml = yaml.load(info_file)

    if info_yaml['map']['pose']['x'] != 0 or info_yaml['map']['pose']['y'] != 0 or info_yaml['map']['pose']['z'] != 0 or info_yaml['map']['pose']['theta'] != 0:
        print_error("convert_stage_map_to_gt_map: map not in origin")

    initial_position_meters = np.array([float(info_yaml['robot']['pose']['x']), float(info_yaml['robot']['pose']['y'])])
    print("initial position (meters):", initial_position_meters)

    map_size_meters = np.array([float(info_yaml['map']['size']['x']), float(info_yaml['map']['size']['y'])])
    print("map_size (meters):", map_size_meters)
    map_size_pixels = np.array(gt.size)
    print("map_size (pixels):", map_size_pixels)
    resolution = map_size_meters / map_size_pixels * np.array([1, -1])  # meter/pixel, on both axis, except y axis is inverted in image
    print("resolution:", resolution)

    c_x, c_y = map_center_pixels = map_size_pixels / 2
    print("map center (pixels):", map_center_pixels)
    p_x, p_y = initial_position_pixels = map(int, map_center_pixels + initial_position_meters / resolution)
    print("initial position (pixels):", initial_position_pixels)

    if pixels[p_x, p_y] != (205, 205, 205):
        print_fatal("initial position in a wall pixel")

        new_initial_position_found = False
        if p_x != c_x or p_y != c_y:
            line = np.array(xiao_line(p_x, p_y, c_x, c_y))
            sorted_line = sorted(line, key=lambda p_l: np.sum(p_l - np.array([p_x, p_y]))**2)

            for x, y in sorted_line:
                if pixels[x, y] == (205, 205, 205):
                    initial_position_pixels = (x, y)
                    new_initial_position_found = True
                    print_error("found free pixel moving toward the center of the map in {}. Output image needs to be manually checked".format(initial_position_pixels))
                    break

        if not new_initial_position_found:
            print_error("initial position can not be found. Skipping image: {}".format(ground_truth_map_file_path))
            return

    # convert to free the pixels accessible from the initial pose
    floodfill(gt, initial_position_pixels, (254, 254, 254), thresh=10)

    if backup_if_exists:
        backup_file_if_exists(ground_truth_map_file_path)

    try:
        print_info("writing to {}".format(ground_truth_map_file_path))
        gt.save(ground_truth_map_file_path)
        gt.save(path.join("/home/enrico/tmp/gt_maps", path.basename(path.dirname(ground_truth_map_file_path))+'.pgm'))
    except IOError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())
    except TypeError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())


def compute_ground_truth_from_stage_map_for_all(all_datasets_folder):
    stage_world_folders = sorted(map(path.abspath, map(path.dirname, glob.glob(path.join(all_datasets_folder, "**/environment.world")))))
    for stage_world_folder in stage_world_folders:
        compute_ground_truth_from_stage_map(stage_world_folder, do_not_recompute=False, backup_if_exists=True)


if __name__ == '__main__':
    compute_ground_truth_from_stage_map_for_all(path.expanduser("~/ds/performance_modelling_all_datasets"))
    # compute_ground_truth_from_stage_map(stage_world_folder=path.expanduser("~/ds/performance_modelling_all_datasets/test"), do_not_recompute=False, backup_if_exists=True)
