# Simulation of real-time gait detectiong with modified Kheirkhahan method

There are two potential approaches presented here: first one using the data recorded from a single wrist to classify the walking directly for the wrist, and the second using the data from two wrists to make the gait detection prediction. 

To see the performance of the two methods, run the scripts `evaluate_per_wrist.py` and `evaluate_fused.py` to respectively see the metrics of the algorithm performed either from a single wirst data or with the data from two wrists fused. 

Both approaches address time discountinueties by segmenting the data at those discountinuities and ensuring that the windows are not passed over the junction of the segments. The discountinueties are found by only examining increasing timestamps and if the gap between the timestamps is sufficiently big, a new segment is assigned in the dataframe. 

## Single wrist approach 
To run the single wrist approach, run the following command where `PATH-TO-FIlE`should be replaced with the path to the txt file with the data to be evaluated:
```bash
evaluate_per_wrist.py PATH-TO-FILE
```
The following steps are made in the script:

* The file from the provide directory is loaded and segmented into fragments of non-faulty data.
* The acceleration columns are found and translated to m/s<sup>2</sup>
* The annotations are read off from the file. 
* The algorithm is applied in a manner to simulated real-time application: it is applied on 13s from the data at a time, where only the middle 9s are evaulated with 2s buffer at the start and at the end. The gait prediction are then recorded. 
* The metrics and the plot for the predictions compared to the true label are presented. 

Some of the funtions are taken from `detect_per_wrist.py` file, which aims to run only the parts of the same procedure which to not require the prior knowledge of the true labels. 


## Fused wrist approach 