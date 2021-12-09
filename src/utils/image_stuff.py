import cv2
import numpy as np
from matplotlib import pyplot as plt


class image_stuff:

    # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    def resize_and_pad(image, desired_size, PADDING=False):

        old_size = image.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        new_image = cv2.resize(image, (new_size[1], new_size[0]))
        left, top, right, bottom = [0, 0, 0, 0]
        if PADDING:
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            new_image = cv2.copyMakeBorder(
                new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )
        return new_image, ratio, [left, top, right, bottom]

    def draw_bbox(image, bbox, class_name="", box_color=(255, 0, 0), thickness=2):
        """draw a single bounding box on the image"""
        out_image = image.copy()
        x_min, y_min, x_max, y_max = (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )

        cv2.rectangle(
            out_image,
            (x_min, y_min),
            (x_max, y_max),
            color=box_color,
            thickness=thickness,
        )

        fontScale = 0.35
        ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * fontScale, 1
        )
        cv2.rectangle(
            out_image,
            (x_min, y_min - int(text_height)),
            (x_min + text_width, y_min),
            box_color,
            -1,
        )
        cv2.putText(
            out_image,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )
        return out_image

    def draw_bbox_form_contour(
        image,
        contour,
        rectangle_color=(0, 255, 0),
        rectangle_thickness=5,
        contour_color=(255, 0, 0),
        contour_thickness=2,
    ):
        x, y, w, h = cv2.boundingRect(contour)
        output_image = cv2.rectangle(
            image.copy(),
            (x, y),
            (x + w, y + h),
            rectangle_color,
            rectangle_thickness,
        )

        cv2.drawContours(output_image, [contour], -1, contour_color, contour_thickness)
        return output_image

    def display_image(image, window_name, FULL_SCREEN=False):

        if FULL_SCREEN:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)

    def plt_display_image(image, title="", colormap="viridis", figsize=(15, 15)):
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=colormap)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

    # https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv/50757596
    def change_brightness(image, alpha, beta):
        return cv2.addWeighted(
            image, alpha, np.zeros(image.shape, image.dtype), 0, beta
        )

    def fix_illumination(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv[..., 2].mean() < 50:
            alpha = int(75 / hsv[..., 2].mean())
            eq_image = image_stuff.change_brightness(image, alpha, 0)

        elif hsv[..., 2].mean() > 125:
            alpha = 0.7
            eq_image = image_stuff.change_brightness(image, alpha, -25)

        else:
            eq_image = image.copy()
        return eq_image

    def find_eye_area(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        threshold_image = cv2.morphologyEx(
            threshold_image, cv2.MORPH_OPEN, kernel, iterations=5
        )
        (r, c) = np.where(threshold_image > 10)

        min_row, min_col = np.min(r), np.min(c)
        max_row, max_col = np.max(r), np.max(c)

        width = max_col - min_col
        height = max_row - min_row

        if min_row == 0:
            height += min_col * 2

        eye_area = height * width
        return eye_area

    def crop_eye_area(image, pad_x=0, pad_y=0):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        threshold_image = cv2.morphologyEx(
            threshold_image, cv2.MORPH_OPEN, kernel, iterations=5
        )
        (r, c) = np.where(threshold_image > 10)
        min_row, min_col = np.min(r), np.min(c)
        max_row, max_col = np.max(r), np.max(c)

        cropped_image = image.copy()
        cropped_image = cropped_image[
            min_row + pad_y : max_row - pad_y, min_col + pad_x : max_col - pad_x
        ]

        return cropped_image
