import os 
import xml.etree.ElementTree as ET

def convert_box(size, box):
    """
    Convert bounding box coordinates from VOC format to YOLO format.
    
    VOC Format: (xmin, xmax, ymin, ymax)
    YOLO Format: (x_center_normalized, y_center_normalized, width_normalized, height_normalized)

    Parameters:
    - size: tuple (image width, image height)
    - box: bounding box in VOC format

    Returns:
    - Bounding box in YOLO format
    """
    
    # Calculate normalization factors for width and height
    dw, dh = 1. / size[0], 1. / size[1]
    
    # Convert VOC coordinates to YOLO format
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    
    return x * dw, y * dh, w * dw, h * dh

def convert_voc_to_yolo():
    """
    Convert VOC XML annotations to YOLO format text files.
    """
    
    # Iterate through all annotations in the 'labels' directory
    for anno in os.listdir('./data/labels'):
        # Check if the file is an XML file
        if anno.split('.')[1] == 'xml':
            file_name = anno.split('.')[0]
            
            # Open a new text file for writing YOLO formatted annotations
            out_file = open(f'./data/labels/{file_name}.txt', 'w')
            
            # Parse the XML file
            tree = ET.parse(os.path.join('data','labels', anno))
            root = tree.getroot()
            
            # Extract image width and height from the XML
            size = root.find('size')        
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            # Define classes
            names = ['trafficlight', 'speedlimit', 'crosswalk', 'stop']
            
            # Iterate through each object/annotation in the XML
            for obj in root.iter('object'):
                cls = obj.find('name').text
                
                # Check if the class of the object is in our defined classes and is not marked as 'difficult'
                if cls in names and int(obj.find('difficult').text) != 1:
                    # Extract bounding box in VOC format
                    xmlbox = obj.find('bndbox')
                    bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                    
                    # Get class ID
                    cls_id = names.index(cls)
                    
                    # Write to the output file
                    out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')
