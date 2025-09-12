import copy
import os

import numpy as np
from aicspylibczi import CziFile
from pathlib import Path

import xml.etree.ElementTree as ET
import xml.dom.minidom

import matplotlib.pyplot as plt
import tifffile
from pprint import pprint
import h5py
import sys




czi_0_24_hours = '2024-12-21-PRRX1-MYOD-3-01.czi'
base_pth = Path("/scratch/indikar_root/indikar1/shared_data/imaging_test/full_CZIs")
pth = base_pth / czi_0_24_hours
czi_0_24 = CziFile(pth)


czi_24_72_hours = '2024-12-22-PRRX1-MYOD-3-02.czi'
pth = base_pth / czi_24_72_hours 
czi_24_72 = CziFile(pth)
czi = czi_0_24


# Each scene has its own file.
def save_scene_to_hdf5(scene_array, scene, count, timepoint):
    """
    Convert scene array with basic metadata into HDF5 file
    """
    data = scene_array
    
    dt = h5py.string_dtype(encoding='utf-8')  
    # Saving to HDF5
    with h5py.File(f"{scene}_{count}.hdf5", "w") as f:
        dset = f.create_dataset("Scene", data=data, compression="gzip")
        
        grp = f.create_group('metadata')
        grp.attrs['sample'] = scene

        f.create_dataset("channel_names", data=np.array(["tagGFP", "MKate", "Cy5"], dtype=dt))


def save_frame_to_hdf5(frame, scene, count, timepoint, basepath):
    """
    Save a single frame into a HDF5 file
    """
    data = frame
    
    dt = h5py.string_dtype(encoding='utf-8')  
    # Saving to HDF5
    # bpath = f"/nfs/turbo/umms-indikar/Ram/projects/reprogramming/Image_analysis/convertor/{scene}/{count}"
    bpath = f"{basepath}/{scene}/{count}"
    os.makedirs(bpath, exist_ok=True)
    with h5py.File(f"{bpath}/t_{timepoint}.hdf5", "w") as f:
        dset = f.create_dataset("Scene", data=data, compression="gzip")
        
        grp = f.create_group('metadata')
        grp.attrs['sample'] = scene
        grp.attrs['sampleID'] = count
        grp.attrs['time'] = timepoint

        f.create_dataset("Channels", data=np.array(["tagGFP", "MKate", "Cy5", "Oblique"], dtype=dt))


def proc_czi(czi, scene, scene_name, counter, timedelay):

    metadata = czi.meta
    my_dims = czi.dims
    my_size = czi.size
    dimensions = czi.get_dims_shape()
    print(my_dims)
    print(dimensions)
    
    x = dimensions[0]['X'][1]
    y = dimensions[0]['Y'][1]
    
    timepoints = dimensions[0]['T'][1] 
    channels = dimensions[0]['C'][1]
    mosaic = dimensions[0]['M'][1]
    # rows = 6
    # columns = 5
    
    # channels = 3
    rows = 4
    columns = 4

    # for some stitches the row order gets flipped in the software. so the ordering should be reversed in alternating rows
    invert = True
    xoverlap = 22 # The number of pixels that overlap on both sides
    yoverlap = 27
    
    print(f'Timepoints: {timepoints}')
    print(f'Channels: {channels}')
    print(f'Rows: {rows}')
    print(f'Columns: {columns}')
    print(f'X: {x}')
    print(f'Y: {y}')
    print(f'Mosaic: {mosaic}')
    

    
    timeframes = []
    # Iterate timepoints collecting each frame
    for t in range(timepoints):
        multi_channel = []
        # Iterate channels
        for c in range(channels):
            full_image = []
            # Iterating through each mosaic segment
        
            for i in range(rows):
                for j in range(columns):
                    
                    if i%2 == 1 and invert:
                        # jind = ((i*j) - j)
                        pos = (columns * i) + (columns - j) - 1
                    else:
                        pos = (columns * i) + j

                    img, shp = czi.read_image(C=c, S=scene, Z=0, T=t, M=pos)
                    proc_img = img[0, 0, 0, 0, 0, 0, ::, ::]
                    proc_img = proc_img[xoverlap:-xoverlap, yoverlap:-yoverlap]
                
                    full_image.append(proc_img)
                        
            grid = np.empty(rows * columns, dtype=object)
            grid[:] = full_image
            
            grid = grid.reshape(rows, columns)

            vertical = []
            for i in range(rows):
                horizontal = []
                for j in range(columns):
                    horizontal.append(grid[i,j])
                vertical.append(np.hstack(horizontal))
            stitched = np.vstack(vertical)

            # another way to stack the grid
            # stitched = np.vstack([
            #     np.hstack([grid[0, 0], grid[0, 1]]),
            #     np.hstack([grid[1, 0], grid[1, 1]])
            # ])
            multi_channel.append(stitched)
        frame = np.stack(multi_channel, axis=0)
        # save_frame_to_hdf5(frame, scene_name, counter, timedelay + t)
        return frame, metadata



scenemap = {"Negative_Controls": [1, 7, 13], "Myod": [2, 3], "PRRX1": [14, 15], "Myod_PRRX1": [8, 9, 11]}

if __name__ == '__main__':
    
    args = sys.argv
    scene_name, scene, counter = args[1], int(args[2]), int(args[3])


    print("Running Scene", scene_name, counter)
    proc_czi(czi_0_24, scene, scene_name, counter, timedelay = 0)
    proc_czi(czi_24_72, scene, scene_name, counter, timedelay = 67)
    print("saving Scene", scene_name, counter)