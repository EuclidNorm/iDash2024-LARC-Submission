# iDash Federated Learning Competition

This tool is built on FLamby. For detailed documentation and information, please visit [FLamby GitHub page](https://github.com/owkin/FLamby).

## Table of Contents
- [Dependencies](#dependencies)
- [Dataset Description](#description)
- [Using the Dataset](#dataset)
- [Run Federated Learning](#federated)
- [Task](#task)

## Dependencies
FLamby requires Python 3. You need to install the additional Python libraries listed in ```requirements.txt``` to run FLamby. To install these libraries, run:
```
pip3 install -r requirements.txt
```

## Dataset Description

Preprocessed data is stored in this repo in the file ```flamby/datasets/fed_tcga_brca/brca.csv```, so the dataset does not need to be downloaded. These data are collect from 6 medical centers (0-5), and they are randomly splited into training group and testing group. Here is the number of dataset in each center.

|                      | Center 0 | Center 1 | Center 2 | Center 3 | Center 4 | Center 5 |
|----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:
| No. of Training Data | 279 | 165 | 174 | 131 | 131 | 20

The data for all 6 centers is presented in this format, and each data has 42 columns. You can view the raw data at ```flamby/datasets/fed_tcga_brca/brca.csv```

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Clinical data from the TCGA-BRCA study with 1,088 patients.
| Dataset size       | 117,5 KB (stored in this repository).
| Centers            | 6 regions - Northeast, South, West, Midwest, Europe, Canada.
| Inputs shape       | 39 features (tabular data).
| Targets shape      | (E,T). E: relative risk, continuous variable. T: event observed (1) or censorship (0)
| Total nb of points | 900.


## Using the Dataset
Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_tcga_brca import FedTcgaBrca

# To load the first center as a pytorch dataset
center0 = FedTcgaBrca(center = 0)
# To load the second center as a pytorch dataset
center1 = FedTcgaBrca(center = 1)
```

You can execute the ```getdata.py``` file to load the data in a specific data center. Run
```
python3 getdata.py
```

The script will display the data in center 0, which has 279 training data points as described in the previous table.

## Run Federated Learning
We provide a sample program to train the dataset in all centers using Federated Learning Adam. The sample script is ```federated.py```, and you can run the program with
```
python3 federated.py
```

This program includes the training section, providing an end-to-end demonstration of how to run federated learning with this tool.

The script defines the train data loader in variable ```train_dataloaders```. ```train_dataloaders``` is a ```list``` with ```torch.utils.data.DataLoader```. We add all six centers into ```train_dataloaders```.

After defining ```train_dataloaders```, another dictionary variable ```args``` is initialized. This dictionary includes the previously defined ```train_dataloaders``` and other training parameters.

## Task
You need to develop a new federated learning algorithm for this competition. There is no need to modify any part of the tool or data reading process, as everything has already been set up and is ready to use.

The demonstration federated learning algorithm is defined in the ```solution``` function within the ```todo.py``` file. You need to rewrite this function with your algorithm. Then, you can test your algorithm with
```
python3 federated.py
```

```solution``` function takes one input parameter ```local_updates```, which is a list with ```n``` items. Since the dataset has 6 centers, there are 6 items in ```local_updates```, and each representing data in one of the 6 centers. Each item is a dictionary in the following format where ```n_samples``` is the number of data in each center
```
{
    "updates": updates, 
    "n_samples": size
}
 ```
```updates``` is a list of ```numpy.ndarray```, and the total number of data in all centers can be calculated by ```n_samples```. You can view more details about the ```local_updates``` definition in the ```calc_aggregated_delta_weights``` method located in ```flamby/strategies/fed_opt.py```, from lines 168 to 192.

The iDash Track 2 challenge focuses on developing an efficient method for calculating aggregation weights. Therefore, please modify both the `todo.py` and `fed_opt.py` files to adjust the aggregation weight calculation.

Currently, the `todo.py` script uses the number of samples from each client as the aggregation weight. You may choose to build upon this approach or disregard the number of samples entirely. Additionally, you are allowed to introduce up to three new statistical metrics on the client side. These metrics should each represent a single numerical value derived from the client’s dataset, such as the mean, variance, etc.

iDash's final evaluation will involve running the modified `todo.py` and `fed_opt.py` files on a test dataset to assess their performance.
