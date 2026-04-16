# Simulation of real-time gait detectiong with modified Kheirkhahan method

There are two potential approaches presented here: first one using the data recorded from a single wrist to classify the walking directly for the wrist, and the second using the data from two wrists to make the gait detection prediction. 

The detection of gait from the two algorithms can be found in files `detect_per_wrist.py` and `detect_fused.py`. These do not assume the existance of labels in the data and instead just open the file and produce predictions of walk from the data in a real-time manner. 

To see the performance of the two methods, run the scripts `evaluate_per_wrist.py` and `evaluate_fused.py` to respectively see the metrics of the algorithm performed either from a single wirst data or with the data from two wrists fused. 

Both approaches address time discountinueties by segmenting the data at those discountinuities and ensuring that the windows are not passed over the junction of the segments. The discountinueties are found by only examining increasing timestamps. Further if the gap between the timestamps is sufficiently big, a new segment is assigned in the dataframe. 

The main algorithm for both of the methods is provided in `GSD3.py` and it utilized `ActivityCounts.py` script. The method also builds upon the scripts provided in [multimob.GSD.utils](https://github.com/DMegaritis/multimobility_wrist/tree/main/multimob/GSD/utils). The following packages are used: 

* pandas
* numpy 
* typing_extensions
* typing
* scipy
* sys (for accessing the file)
* os (for accessing the file)
* csv (for accessing the file)
* collections
* datetime
* warnings
* matplotlib.pyplot (for plotting the results)
* matplotlib.ticker (for plotting the results)
* sklearn.metrics (assessing the metrics)

## Single wrist approach 
To run the single wrist approach, run the following command where `PATH-TO-FIlE`should be replaced with the path to the txt file with the data to be evaluated:
```bash
python -m models.realtime.evaluate_per_wrist.py PATH-TO-FILE
```
The following steps are made in the script:

* The file from the provide directory is loaded and segmented into fragments of non-faulty data.
* The acceleration columns are found and translated to m/s<sup>2</sup>
* The annotations are read off from the file. 
* The algorithm is applied in a manner to simulated real-time application: it is applied on 13s from the data at a time, where only the middle 9s are evaulated with 2s buffer at the start and at the end. The gait prediction are then recorded. 
* The metrics and the plot for the predictions compared to the true label are presented. 

Some of the funtions are taken from `detect_per_wrist.py` file, which aims to run only the parts of the same procedure which to not require the prior knowledge of the true labels. 


## Fused wrist approach 

To run the fused approach, execute the commanf in the same way but with the provided path, `PATH-TO-FOLDER`, leading to the directory containing both files for the right and the left wirst. The file names of those files are expected to be `s1_1RW.txt` and `s2_2LW.txt`. 
```bash 
python -m models.realtime.evaluate_per_wrist.py PATH-TO-FOLDER
```
For this approach the following steps are taken: 

* The data from both wrist files is loaded in the same way as before. The acceleration values are again translated to m/s<sup>2</sup>. 
* The two data frames are fused into one. 
* Both of the files are simulated in "real-time" manner, with taking 13s at a time to evaluate. 
* The gait is detected if both wrists show that it is present. In cases where the data is absent (NaN) in one of the wrists, the prediction is taken from one wrist only. 
* The metrics and the plot are present in the manner similar to the first approach. 

The file `detect_fused.py` again provides only the functions that do not rely on the knowledge of the annotations. 