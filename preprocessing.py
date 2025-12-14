import cv2
import pandas as pd
import numpy as np
import image_processing


def open_csvfile(name): #opens and does initial processing on raw data
    df = pd.read_csv(name)
    #print(df.head())
    #df.columns = df.iloc[0] # set headers to first row
    #df = df[1:] # drop row w/ headers
    df = df.drop('s1', axis=1) #remove intermed strength values
    df = df.drop('s2', axis=1)
    df = df.drop('s3', axis=1)
    df = df.drop('Strength', axis=1)

    df['x'] = np.nan
    df['y'] = np.nan
    #df['z'] = np.nan
    df['size_x'] = np.nan
    df['size_y'] = np.nan

    print(df.columns)
    
    return df

def groupby_climb(df): #groups climbs in the dataframe into a dict with keys (climb code, date)
    df = df.groupby(["Climb", "Date"])
    return df, list(df.groups.keys())   

def preprocess_images(folder_path):
    image_processing.overlay_folder(folder_path)

def process_climbs(df, g_df, c_labels, foldername): #processes images and associates correct manually-taken data with coordinates. dataframe (groupby), list, string
    print("\n","~"*10)
    print(f"processing climbs in {foldername}\n")
    
    imagedata = image_processing.process_images(foldername)

    climbs_imageprocessed = imagedata.keys()

    #climbs_imageprocessed = [((i_spl:=i.split("~"))[1], "/".join(i_spl[0].split("_"))) for i in climbs_imageprocessed]
    print(climbs_imageprocessed)

    climb_labels = set(c_labels)&set(climbs_imageprocessed)

    print("\tmissing image data for", set(c_labels)-climb_labels)
    print("\tmissing manual data for", set(climbs_imageprocessed)-climb_labels)

    climb_labels = list(climb_labels)

    print("\nlabels being processed: ",climb_labels)

    group_indices = g_df.indices

    for label in climb_labels:
        process_climb(df, list(group_indices[label]), imagedata[label])

    print(".\n.\n.\nSuccessfully processed.\n","~"*10)

    df.to_csv("all_data_wc.csv")

    return df

def process_climb(df, climb_indices, blocks):
    for i, block in enumerate(blocks):
        i = climb_indices[i]
        df.loc[i,"x"] = block["x"]
        df.loc[i,"y"] = block["y"]
        df.loc[i,"size_x"] = block["size_x"]
        df.loc[i,"size_y"] = block["size_y"]

    #print(df[climb_indices[0]:climb_indices[-1]+1]) #debug: print rows that should have been updated

    return df

def process_data(image_folder_path):
    data = open_csvfile("all_data.csv")

    grouped_data, climb_labels = groupby_climb(data)

    #image_folder_path = "G:\My Drive\Alwood-SeniorResearch\Code\Images" 
    data = process_climbs(data, grouped_data, climb_labels, image_folder_path)

    return data


def create_graph(df, climb): 
    graph = {}
    return graph

process_data("G:\My Drive\Alwood-SeniorResearch\Code\Images")