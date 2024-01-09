#!pip install ultralytics
import os
import csv
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
#folder delete
import shutil

import pandas as pd

%cd /content
##to find the pixel and feet
img1 = cv2.imread("/content/redline1.jpg")

lowcolor1 = (0, 0, 200)
highcolor1 = (50, 50, 255)
thresh = cv2.inRange(img1, lowcolor1, highcolor1)
#cv2.imwrite('red_line_thresh1.png', thresh)--
#cv2_imshow(thresh)--

# Apply morphology close
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Get contours and filter on area
result = img1.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
result = img1.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 5000:
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

# Save resulting images
cv2.imwrite('red_line_thresh11.png', thresh)
#cv2.imwrite('red_line_extracted1.png', result)--

# Read the thresholded image
#img = cv2.imread('red_line_thresh11.png')--

# Load the binary mask image
mask = cv2.imread('red_line_thresh11.png', cv2.IMREAD_GRAYSCALE)

# Define the color value for white
white_color = 255

# Find the coordinates of all white pixels in the image
white_pixels = np.column_stack(np.where(mask == white_color))

# Check if there are any white pixels in the image
if len(white_pixels) > 0:
    # Find the leftmost and rightmost coordinates
    leftmost_coordinate = tuple(white_pixels[white_pixels[:, 1].argmin()])
    rightmost_coordinate = tuple(white_pixels[white_pixels[:, 1].argmax()])

    # Calculate the length in pixels
    length_in_pixels = rightmost_coordinate[1] - leftmost_coordinate[1]

    #print(f"Leftmost Coordinate: {leftmost_coordinate}")
   # print(f"Rightmost Coordinate: {rightmost_coordinate}")
    #print(f"Length in Pixels: {length_in_pixels}")
#else:
    #print("No white pixels found in the image.")


# Open the text file for reading
with open("IMG15_len.txt", "r") as file:
    # Read the single line from the file
    line = file.readline()

# Extract the numeric part from the line
numeric_value = ''.join(char for char in line if char.isdigit() or char == '.')

# Check if a numeric value was found
if numeric_value:
    numeric_value = float(numeric_value)  # Convert to float if you want to work with numbers
    #print(f"Extracted numeric value: {numeric_value}")
#else:
    #print("No numeric value found in the line.")


#getting the feet by pixel
length= numeric_value/length_in_pixels
print(length)
## above code we have attained the feet/pixel

# Define the path for the CSV file and the folder for chipped images
HOME= os.getcwd()
csv_file_path = os.path.join(HOME, 'object_detection_results.csv')
chipped_image_folder = os.path.join(HOME, 'Chipped_image')

# Define the path for the text file
txt_file_path = os.path.join(HOME, 'Log detected.txt')
# Define the path for the number of wood logs whose circumference is less than 12 inches
txt_path_logSec1 = os.path.join(HOME, 'Log_less_than_12.txt')
# Define the path for the number of wood logs whose circumference is greater than equal 12 inches
#and less than equal 38 inches
txt_path_logSec2 = os.path.join(HOME, 'Log_from_12_to_38.txt')
#Define the path for the number of wood logs whose circumference is greater than 38 inches
txt_path_logSec3 = os.path.join(HOME, 'Log_greater_than_38.txt')


# Ensuring the chipped_images folder exists
os.makedirs(chipped_image_folder, exist_ok=True)

# Load the YOLO model
model = YOLO(f'/content/detect.pt')

# Perform object detection
results = model.predict(source='/content/redline1.jpg', conf=0.25, save=True)
cnt = 0
lessThan12=0
from12To38=0
greaterThan38=0

# Create and open the CSV file for writing
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    for indexx, numx in enumerate(results):
        num_woods = len(results[indexx].boxes.xyxy)

        # Write the column headers
        csvwriter.writerow(["Index", "left top", "bottom right", "Horizontal diameter", "Vertical diameter", "Diameter_pixel","Diameter_feet","Diameter_inches","Circumference_feet","Circumference_inches"])

        for indexy, numy in enumerate(results[indexx].boxes.xyxy):
            box = results[indexx].boxes.xyxy[indexy]

            # Extract coordinates and calculate dimensions
            left_top = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            horizontal_diameter = float(box[2] - box[0])
            vertical_diameter = float(box[3] - box[1])
            diameter_pixel = (horizontal_diameter + vertical_diameter) / 2
            diameter_feet=diameter_pixel* length
            diameter_inches=diameter_feet*12
            circumference_feet= 2*3.14*(diameter_feet/2)
            circumference_inches=2*3.14*(diameter_inches/2)

            # Increment the count
            cnt += 1

            # Writing the data to the CSV file
            csvwriter.writerow([cnt, left_top, bottom_right, horizontal_diameter, vertical_diameter, diameter_pixel, diameter_feet,diameter_inches,circumference_feet,circumference_inches])

            # Crop and save the chipped image
            image = Image.open('/content/redline1.jpg')
            chipped_image = image.crop((left_top[0], left_top[1], bottom_right[0], bottom_right[1]))

            # Convert the image to 'RGB' mode before saving as JPEG
            chipped_image = chipped_image.convert('RGB')

            chipped_image.save(os.path.join(chipped_image_folder, f"{cnt}.jpg"))
            if circumference_inches<12:
              lessThan12=lessThan12+1
            elif  circumference_inches>=12 and circumference_inches<=38:
              from12To38=from12To38+1
            else:
              greaterThan38=greaterThan38+1

            

# Write the value of cnt to the text file
with open(txt_file_path, 'w') as txtfile:
    txtfile.write(str(cnt))
#Write the values of circumference_cnt of 3 diff section to text file 
with open(txt_path_logSec1 , 'w') as txtfile:
    txtfile.write(str(lessThan12))
with open(txt_path_logSec2 , 'w') as txtfile:
    txtfile.write(str(from12To38))
with open(txt_path_logSec3 , 'w') as txtfile:
    txtfile.write(str(greaterThan38))        
   

#print("CSV file created at:", csv_file_path)
#print(f"Chipped images saved in the folder: {chipped_image_folder}")
#print(f"Value of cnt ({cnt}) saved in the text file: {txt_file_path}")

#defect
# Save the current working directory
current_directory = os.getcwd()

# Change to the desired directory
%cd /content/defects
print("Current Directory:", current_directory)

import torch
import utils

#deleting if the exp folder already contains
def delete_exp_directory():
    exp_path = '../defects/runs/predict-cls/exp'

    # Check if the directory exists
    if os.path.exists(exp_path):
        # Remove the directory and its contents
        shutil.rmtree(exp_path)
# Call the function to delete the directory if it exists
delete_exp_directory()

!python classify/predict.py --weights /content/defects/runs/best.pt --img 128 --source /content/Chipped_image --save-txt

%cd /content
def process_text_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as file:
        first_row = file.readline().strip().split(' ')

    # Extract the numeric part of the file name without the '.txt' extension
    file_index = os.path.splitext(file_name)[0]

    csv_data = {'Name': file_index, 'Defect': first_row[1]}

    return csv_data

folder_path = '/content/defects/runs/predict-cls/exp/labels'  # Replace with the actual folder path
output_csv_path = '/content/output.csv'  # Replace with the desired output CSV file path

with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['Name', 'Defect']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header
    writer.writeheader()

    # Process and sort each text file in the folder
    files_to_process = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]
    sorted_files = sorted(files_to_process, key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in sorted_files:
        csv_data = process_text_file(folder_path, file_name)
        writer.writerow(csv_data)

print("CSV file created successfully.")

# Load the original CSV file
original_csv_path = "/content/object_detection_results.csv"
original_df = pd.read_csv(original_csv_path)

# Load the CSV file with the "Defect" column
output_csv_path = "/content/output.csv"
output_df = pd.read_csv(output_csv_path)

# Add the "Defect" column to the original DataFrame
original_df['Defect'] = output_df['Defect']

# Save the modified DataFrame back to the original CSV file
original_df.to_csv(original_csv_path, index=False)

#print("Defect column added to object_detection_results.csv successfully.")


#log_types
# Save the current working directory
current_directory = os.getcwd()

# Change to the desired directory
%cd /content/type_class
print("Current Directory:", current_directory)

import torch
import utils

#deleting if the exp folder already contains
def delete_exp_directory():
    exp_path = '../type_class/runs/predict-cls/exp'

    # Check if the directory exists
    if os.path.exists(exp_path):
        # Remove the directory and its contents
        shutil.rmtree(exp_path)
# Call the function to delete the directory if it exists
delete_exp_directory()

!python classify/predict.py --weights /content/type_class/runs/best.pt --img 128 --source /content/Chipped_image --save-txt

%cd /content
def process_text_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r') as file:
        first_row = file.readline().strip().split(' ')

    # Extract the numeric part of the file name without the '.txt' extension
    file_index = os.path.splitext(file_name)[0]

    csv_data = {'Name': file_index, 'Type': first_row[1]}

    return csv_data

folder_path = '/content/type_class/runs/predict-cls/exp/labels'  # Replace with the actual folder path
output_csv_path = '/content/output1.csv'  # Replace with the desired output CSV file path

with open(output_csv_path, 'w', newline='') as csv_file:
    fieldnames = ['Name', 'Type']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header
    writer.writeheader()

    # Process and sort each text file in the folder
    files_to_process = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.txt')]
    sorted_files = sorted(files_to_process, key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in sorted_files:
        csv_data = process_text_file(folder_path, file_name)
        writer.writerow(csv_data)

print("CSV file created successfully.")

# Load the original CSV file
original_csv_path = "/content/object_detection_results.csv"
original_df = pd.read_csv(original_csv_path)

# Load the CSV file with the "Defect" column
output_csv_path = "/content/output1.csv"
output_df = pd.read_csv(output_csv_path)

# Add the "Defect" column to the original DataFrame
original_df['Type'] = output_df['Type']

# Save the modified DataFrame back to the original CSV file
original_df.to_csv(original_csv_path, index=False)

#print("Type column added to object_detection_results.csv successfully.")