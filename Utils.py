import os
import cv2
import xml.etree.ElementTree as ET
import shutil

class Utils:
    def save_all_frames(video_path, output_folder):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            frame_count += 1
            frame_filename = os.path.join(output_folder, f'{frame_count}.jpg')  # Frame naming convention
            
            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)
            
        print('All frames saved!')
        cap.release()
        cv2.destroyAllWindows()
            
    def get_frame(sec, video):
        video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frames, image = video.read()
        return has_frames, image

    def draw(frame, bbox):
        x, y, w, h = map(int, bbox)
        nest_box = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)
        return nest_box

    def get_bounding_box(video_path):
        # Select bounding box
        sec = 0
        vidcap = cv2.VideoCapture(filename=video_path)
        success, frame = Utils.get_frame(sec, vidcap)

        bbox = cv2.selectROI("Select bounding box and press SPACE", frame, showCrosshair=False)

        vidcap.release()
        cv2.destroyAllWindows()

        return bbox

    # Function to convert XML annotations to YOLO format
    def convert_xml_to_yolo(xml_file, class_dict):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_dict:
                continue  # Skip classes not in class_dict
            class_id = class_dict[class_name]

            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)

            yolo_labels.append(f"{class_id} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}")

        return yolo_labels
    
    def copy_images_labels(dataset_dir, images_dir, labels_dir):
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        subdirs = [os.path.join(dataset_dir, subdir) for subdir in os.listdir(dataset_dir) 
                   if os.path.isdir(os.path.join(dataset_dir, subdir))]
        
        for subdir in subdirs:
            for file in os.listdir(subdir):
                dir_name = subdir.split('\\')[-1]
                if file.endswith('.PNG'):
                    shutil.copy(os.path.join(subdir, file), os.path.join(images_dir, f'{dir_name}_{file}'))

                elif file.endswith('.txt'):
                    shutil.copy(os.path.join(subdir, file), os.path.join(labels_dir, f'{dir_name}_{file}'))
                 

    # Function to parse all XML files in a directory and convert to YOLO format
    def convert_xml_dir_to_yolo(xml_dir, images_dir, labels_dir, class_dict):
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for xml_file in os.listdir(xml_dir):
            if xml_file.endswith('.jpg'):
                 image_name = os.path.splitext(xml_file)[0]
                 shutil.copy(os.path.join(xml_dir, xml_file), os.path.join(images_dir, xml_file))
                 with open(os.path.join(labels_dir, f"{image_name}.txt"), 'a') as f:
                    pass 
                    
            if xml_file.endswith('.xml'):
                image_name = os.path.splitext(xml_file)[0]
                yolo_labels = Utils.convert_xml_to_yolo(os.path.join(xml_dir, xml_file), class_dict)

                # Write YOLO labels to output file
                with open(os.path.join(labels_dir, f"{image_name}.txt"), 'w') as f:
                    f.write('\n'.join(yolo_labels))  
