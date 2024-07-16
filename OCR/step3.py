import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read bounding boxes from a file (already provided)
def read_bounding_boxes(filename):
    bounding_boxes = []
    with open(filename, 'r') as file:
        for line in file:
            if(line=="\n"):
                continue
            print("PROCESSING LINE: ",line)
            coords = list(map(int, line.strip().split(',')))
            print("PROCESSED: ",coords)
            bounding_boxes.append(coords)
    return bounding_boxes

# Function to check if two bounding boxes intersect
def do_boxes_intersect(box1, box2):
    sizex = 42
    sizey = 3

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
    return not (x2 + sizex < x3 or x4 + sizex < x1 or y2 + sizey < y3 or y4 + sizey < y1)

# Function to merge two intersecting bounding boxes
def merge_boxes(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    new_x1 = min(x1, x3)
    new_y1 = min(y1, y3)
    new_x2 = max(x2, x4)
    new_y2 = max(y2, y4)
    
    return [new_x1, new_y1, new_x2, new_y2]

# Function to merge intersecting bounding boxes
def merge_intersecting_bounding_boxes(bounding_boxes):
    merged_boxes = []
    while bounding_boxes:
        current_box = bounding_boxes.pop(0)
        intersecting_boxes = [current_box]
        
        for other_box in bounding_boxes[:]:
            if any(do_boxes_intersect(box, other_box) for box in intersecting_boxes):
                intersecting_boxes.append(other_box)
                bounding_boxes.remove(other_box)
        
        merged_box = intersecting_boxes[0]
        for box in intersecting_boxes[1:]:
            merged_box = merge_boxes(merged_box, box)
        
        merged_boxes.append(merged_box)
    
    return merged_boxes

# Function to save merged bounding boxes to a file
def save_merged_bounding_boxes(merged_boxes, output_filename):
    with open(output_filename, 'w') as file:
        for box in merged_boxes:
            file.write(','.join(map(str, box)) + '\n')

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), label_prefix=''):
    for i, box in enumerate(bounding_boxes):
        pts = np.array(box).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(image, f'{label_prefix}{i+1}', tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# New function to draw merged bounding boxes on an image
def draw_merged_bounding_boxes(image, merged_boxes, color=(255, 0, 0)):
    for i, box in enumerate(merged_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'M{i+1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Main function to read, merge, save, and draw bounding boxes
def process_and_merge_bounding_boxes(input_filename, output_text_filename, input_image_path, output_image_path):
    bounding_boxes = read_bounding_boxes(input_filename)
    merged_boxes = merge_intersecting_bounding_boxes(bounding_boxes)
    save_merged_bounding_boxes(merged_boxes, output_text_filename)
    
    # Draw merged bounding boxes on the image
    image = cv2.imread(input_image_path)
    image_with_merged_boxes = draw_merged_bounding_boxes(image.copy(), merged_boxes)
    
    # Save and show the result
    cv2.imwrite(output_image_path, image_with_merged_boxes)
    plt.imshow(cv2.cvtColor(image_with_merged_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
process_and_merge_bounding_boxes('result/results_second.txt', 'result/results_third.txt', 'input/in.png', 'input/output_third_image.jpg')
