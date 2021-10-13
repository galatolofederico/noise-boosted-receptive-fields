# noise-boosted-receptive-fields

Repository for the paper [Noise Boosted Neural Receptive Fields]()


## Installation

Clone this repository

```
git clone https://github.com/galatolofederico/noise-boosted-receptive-fields.git && cd noise-boosted-receptive-fields
```

Create a `virtualenv` and install the requirements

```
virtualenv --python=python3.7 env && . ./env/bin/activate
pip install -r requirements.txt
```

## Usage

To train a model run

```
python train.py --model <model> --dataset <dataset>
```

Some arguments that you can use are

|     Argument     |               Type              |          Description          |
|:----------------:|:-------------------------------:|:-----------------------------:|
|      --model     |         "nbrf" or "cnn"         |          Model to use         |
|     --dataset    |  "mnist" or"fmnist" or "kmnist" |         Dataset to use        |
|      --noise     | "dataset" or "batch" or "white" |           Noise type          |
| --noise-position |       "head" or "backbone"      |         Noise position        |
|  --without-noise |                                 |  Without the noise superclass |
| --without-others |                                 | Without the others superclass |
|       --lr       |              float              |         Learning rate         |
|   --batch-size   |               int               |           Batch Size          |


To run a trained model run

```
python interact.py --model <trained_mode> --dataset <dataset>
```


## Contributions and license

The code is released as Free Software under the [GNU/GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license. Copying, adapting and republishing it is not only allowed but also encouraged. 

For any further question feel free to reach me at  [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)