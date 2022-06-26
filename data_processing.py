import os
import xml.etree.ElementTree as ET


train_file_path = "data\\train_zip\\train"

test_file_path = "data\\test_zip"



def data_processing(file_path):
    train_images_path_lists = []
    train_xml_path_lists = []
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for f in list_dir:
        file_name = f
        path_combination = os.path.join(file_path,file_name)
        if "xml" in f:
            train_xml_path_lists.append(path_combination)
        else:
            train_images_path_lists.append(path_combination)
    

    return {
        "imgs":train_images_path_lists,
        "xml":train_xml_path_lists
    }

def VOC_parser(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_label = []
    list_with_image = []
    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find("name").text
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_label.append(label)
        if (len(label) == 0):
            print(xml_file)
    return list_with_all_boxes,list_with_all_label

