# -*- coding: utf-8 -*-
import os
import cv2


class PictureAnnotator:
    def __init__(self, picture_dir, annotation_dir):
        self.picture_dir = picture_dir
        self.annotation_dir = annotation_dir

    def read_annotations(self):
        picture_annotation_dict = {}

        for annotation in os.listdir(self.annotation_dir):
            with open(os.path.join(self.annotation_dir, annotation), 'r') as file:
                lines = file.readlines()
                img_name = lines[0].strip()
                point_lines = lines[1:]

                points = []
                for point_line in point_lines:
                    point = point_line.split(',')
                    point = tuple(map(lambda x: int(float(x)), point))  # 只支持整数坐标
                    points.append(point)

                picture_annotation_dict[img_name] = points

        return picture_annotation_dict

    def annotate_pictures(self):

        picture_annotation_dict = self.read_annotations()

        for picture in os.listdir(self.picture_dir):
            img = cv2.imread(os.path.join(self.picture_dir, picture))
            img_name = picture[:-4]  # .jpg
            points = picture_annotation_dict[img_name]
            self.draw_points(img, points)
            annotated_path = f'{self.picture_dir}_annotated'
            if not os.path.exists(annotated_path):
                os.mkdir(annotated_path)
            cv2.imwrite(os.path.join(annotated_path, f'{img_name}_annotated.jpg'), img)

    def draw_points(self, img, points):
        point_size = 1
        point_color = (0, 0, 255)  # BGR
        thickness = 4

        for point in points:
            cv2.circle(img, point, point_size, point_color, thickness)


if __name__ == '__main__':
    picture_annotator = PictureAnnotator('train_1', 'annotation')
    picture_annotator.annotate_pictures()
