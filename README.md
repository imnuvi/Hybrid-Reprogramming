
This repository contains code to explore, visualize and analyze the [HYPED: Hybrid Gene Expression and imaging dataset](https://www.kaggle.com/datasets/thedoodler/hybrid-imaging-and-genex-dataset-hyb-imagen)


Code is split into Gene expression and image analysis



Image analysis workflow:
- Extraction : Specific part of the image across Time, Scene, Mosaic are extracted
- Clipping and Scaling : The 98th percentile values from the (x, y) shaped matrix are obtained for each channel and rest of the values are clipped to get normalized values. These values are then scaled between 0 - 1.
- Stacking : MKate and TagGFP channels are stacked on each other to get the phase transitions from G1 - S phase
- Gaussian Filtering : Convolves the channel values with a gaussian function to remove noise
- Otsu thresholding : Based on the normalized values from the channel, Otsu method separates the values into two classes - foreground ( cells ) and background ( microscopy imaging specific artefacts )
- Watershed Algorithm : Operates on values to identify different region boundaries between the thresholded objects to separate each island into cells
- Segment mapping : Based on the algorithm, normalize any value in the result to 1 everything else to 0 to create a solid mask image
- Tracking : Bayesian tracking on the segmented objects works by comparing each cell and assigning a probability of movement, calculated based on metrics like centroid and mass

Gene Expression workflow:
- Cells are tagged with three experimental condition barcodes
	- Activation of MYOD1
	- Suppression of PRRX1
	- Activation of MYOD1 + Suppression of PRRX1
- Each cell is tagged with 10x genomics cell specific barcodes
- Single cell sequencing is performed to get reads
- The reads are run through EPI2ME single cell pipeline to get the total number of genes that are expressed from the experiment.
- Cell cycle phases are predicted based on marker genes for each phase
- Trajectory inference is run to place cells on pseudotime and visualized.


