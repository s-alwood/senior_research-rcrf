import cv2
import numpy as np
import os

def overlay(image_path):
    GRID_SPACING_METERS = 0.05
    RED_SQUARE_SIZE_M = 0.2
    savedimage=image_path[:-4]+"_grid.jpg"

    img = cv2.imread(image_path) #open image
    if img is None:
        raise FileNotFoundError("Could not load image.")

    h, w = img.shape[:2] #get size of image
    sw, sh = detect_red_square(img) #detect red square

    px_per_meter_x = sw / RED_SQUARE_SIZE_M #derive pixel to meter ratio
    px_per_meter_y = sh / RED_SQUARE_SIZE_M

    grid_px_x = int(px_per_meter_x * GRID_SPACING_METERS) #convert pixels to meters using ratio
    grid_px_y = int(px_per_meter_y * GRID_SPACING_METERS)

    if not os.path.isfile(savedimage):
        overlay = img.copy() #make a copy of the image to overlay

        for cx in range(0, w, grid_px_x):
            cv2.line(overlay, (cx, 0), (cx, h), (0, 0, 255), 2) #add horizontal lines

        for cy in range(0, h, grid_px_y):
            cv2.line(overlay, (0, cy), (w, cy), (0, 0, 255), 2) #add vertical lines

        output = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
        
        cv2.imwrite( savedimage, output) #save new image with gridlines

    return (savedimage, px_per_meter_x, px_per_meter_y)


def detect_red_square(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 90, 90]) #define red hue ranges
        upper_red1 = np.array([10, 255, 255]) 
        lower_red2 = np.array([170, 90, 90])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2] #get size of image
        candidates = []
        for cnt in contours: #determine which candidate is the red square
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x > w * 0.5 and y > h * 0.5:
                candidates.append((cw * ch, x, y, cw, ch))
        if not candidates:
            raise ValueError("No red square found.")
        _, x, y, cw, ch = max(candidates, key=lambda x: x[0])
        return cw, ch #return dimensions

def overlay_folder(folder_path):
    files_info = []
    try:
        for entry in os.listdir(folder_path): #open folder
            full_path = os.path.join(folder_path, entry) #get path of photo
            if os.path.isfile(full_path):
                print(f"File: {folder_path+ "\\" +entry}, {entry.split(".")}")
                if entry.split(".")[-1]=="jpg" and not "_grid" in entry:  #if folder is a jpg
                    ov_info = overlay(folder_path+ "\\" +entry) #overlay it
                    files_info.append(ov_info)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")

    return ov_info

def find_holdinfo(image_path, ppmx, ppmy):
    x, y = "x", "y"
    boxes, origin = detect_boxes(image_path) # detect boxes in image

    origin = {x:origin[x]/ppmx, y: origin[y]/ppmy}

    print(image_path, "\t", origin)  

    for box in boxes:
        box[x] = (box[x]/ppmx)-origin[x]
        box[y] = (box[y]/ppmy)-origin[y]
        box["size_x"] /= ppmx
        box["size_y"] /= ppmy

    return boxes

def detect_boxes(image_path):  
    img = cv2.imread(image_path) # read in image
    print(image_path)
    origin = -1
    if img is None: #if failed, stop
        print(f"Error: Could not read image {image_path}")
        return []
    dimensions = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to greyscale

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV) # create threshold (only accept pure black)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get contours

    box_positions = []
    #cv2.circle(img, center=(0,0), radius= 15,  color=(counter, 255, counter), thickness=-1)
    for contour in contours: #for each contour
        area = cv2.contourArea(contour) #get area
        x, y, w, h = cv2.boundingRect(contour)
        if area > 500: #if big enough, it's a hold
            w //= 2; h //=2 #convert w and h to size (half of width/height)
            cv2.circle(img, center=(x+w,y+h), radius= 10,  color=(0, 255, 0), thickness=-1) #debugging: draw circle for visualization
            box_positions.append({'x': x+w, 'y': dimensions[0]-(y+h), 'size_x': w, 'size_y': h}) # add hold dictionary to list and continue
        elif area > 100: #if area wasn't big enough for a hold, but isn't small enough to be noise, it's the origin
             #get dimensions    
            #print(f"origin at {x},{y}")
            #cv2.circle(img, center=(x,y), radius= 10,  color=(counter, 255, counter), thickness=-1)
            origin = {'x': x+(w/2), 'y': dimensions[0]-y-h/2} #save origin info
    
    cv2.imwrite(f'{image_path}withcircle.jpg', img) #debugging: save image with circles
    #print("detected boxes")
    #print(len(box_positions), origin)

    if origin == -1:
        print(len(box_positions))
        for bp in box_positions: print(bp)
        exit(0)

    return box_positions, origin

def process_images(folder_path):
    images_info = {}
    try:
        for entry in os.listdir(folder_path): #open folder
            full_path = os.path.join(folder_path, entry) #get path of photo
            if os.path.isfile(full_path):
                #print(f"File: {folder_path+ "\\" +entry}, {entry.split(".")}")
                if entry.split(".")[-1]=="jpg" and not "_grid" in entry:  #if folder is a jpg
                    new_path, ppmx, ppmy = overlay(folder_path+ "\\" +entry) #get its info
                    new_path = new_path.split(".")[0]+"+fill."+new_path.split(".")[1]
                    #print("path ", new_path)
                    #new_path+="+fill"
                    info = find_holdinfo(new_path, ppmx, ppmy)
                    i = entry.split(".")[0]
                    #climbs_imageprocessed = [((i_spl:=i.split("~"))[1], "/".join(i_spl[0].split("_"))) for i in climbs_imageprocessed]
                    images_info[((i_spl:=i.split("~"))[1], "/".join(i_spl[0].split("_")))] = info        

            
    except FileNotFoundError:
        print(f"Failed to execute.")

    return images_info

#image_folder_path = "G:\My Drive\Alwood-SeniorResearch\Code\copytest" 
#a = process_images(image_folder_path)