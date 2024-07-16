import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Function to read bounding boxes from a file
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

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), label_prefix=''):
    for i, box in enumerate(bounding_boxes):
        pts = np.array(box).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(image, f'{label_prefix}{i+1}', tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Function to calculate the distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to group bounding boxes using DBSCAN with distance constraint
def group_bounding_boxes(bounding_boxes, document_width, eps=50, min_samples=1, max_gap_ratio=0.025):
    max_gap = max_gap_ratio * document_width
    large_value = 1e6  # A large value to replace infinity
    centers = [(np.mean([box[i] for i in range(0, len(box), 2)]), np.mean([box[i] for i in range(1, len(box), 2)])) for box in bounding_boxes]
    
    # Create a distance matrix with the distance constraint
    n = len(centers)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distance = euclidean_distance(centers[i], centers[j])
            if distance <= max_gap:
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
            else:
                distance_matrix[i, j] = large_value
                distance_matrix[j, i] = large_value
    
    # Apply DBSCAN with the precomputed distance matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
    labels = clustering.labels_
    return labels

# Function to draw grouped bounding boxes on an image
def draw_grouped_bounding_boxes(image, bounding_boxes, labels):
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise, skip it
            continue
        group_boxes = [box for i, box in enumerate(bounding_boxes) if labels[i] == label]
        x_coords = [coord for box in group_boxes for i, coord in enumerate(box) if i % 2 == 0]
        y_coords = [coord for box in group_boxes for i, coord in enumerate(box) if i % 2 == 1]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'G{label+1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# New function to save grouped bounding box coordinates to a file
def save_grouped_bounding_boxes(bounding_boxes, labels, output_filename):
    unique_labels = set(labels)
    results = []
    
    for label in unique_labels:
        if label == -1:  # Noise, skip it
            continue
        group_boxes = [box for i, box in enumerate(bounding_boxes) if labels[i] == label]
        x_coords = [coord for box in group_boxes for i, coord in enumerate(box) if i % 2 == 0]
        y_coords = [coord for box in group_boxes for i, coord in enumerate(box) if i % 2 == 1]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        results.append([x1, y1, x2, y2])
    
    # Save the results to a file
    with open(output_filename, 'w') as file:
        for result in results:
            file.write(','.join(map(str, result)) + '\n')

# Main function to process the image and bounding boxes
def process_image(image_path, bounding_boxes_path, output_image_path, output_text_path):
    image = cv2.imread(image_path)
    bounding_boxes = read_bounding_boxes(bounding_boxes_path)
    document_width = image.shape[1]
    
    # Draw initial bounding boxes
    image_with_boxes = draw_bounding_boxes(image.copy(), bounding_boxes)
    
    # Group bounding boxes and draw grouped bounding boxes
    labels = group_bounding_boxes(bounding_boxes, document_width)
    image_with_grouped_boxes = draw_grouped_bounding_boxes(image_with_boxes.copy(), bounding_boxes, labels)
    
    # Save grouped bounding boxes to file
    save_grouped_bounding_boxes(bounding_boxes, labels, output_text_path)
    
    # Save and show the result
    cv2.imwrite(output_image_path, image_with_grouped_boxes)
    plt.imshow(cv2.cvtColor(image_with_grouped_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
process_image('input/in.png', 'result/result.txt', 'input/output_image.jpg', 'result/results_second.txt')
