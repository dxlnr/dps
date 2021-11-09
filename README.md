<h1 align="center">
  <b>Digital Product School</b><br>
</h1>

<p align="justify">
This work is done for the Digital Product School in Munich. In order to apply for the position of Artificial Intelligence Engineer, this repository is provided. The challenge is to use the Auto MPG dataset from Kaggle (https://www.kaggle.com/uciml/autompg-dataset) to predict the fuel efficiency of a vehicle using a basic regression with TensorFlow.
</p>

## Installation
Create a new conda environment:
```
conda create -n tensor python=3.8
source activate tensor
```

Download and install the package:
```
git clone https://github.com/frommwonderland/dps.git
cd dps
pip install --upgrade pip
pip install -r requirements.txt
```

## Running
Running the little program by simply
```
python main.py
```

## Results
<p align="justify">
The main goal is to predict the 'mpg' parameter based on various others as fuel efficiency (MPG) is a function of many different other parameters. The first image shows some general information about the data.
</p>
<p align="center">
  <img width="85%" height="200" src="https://github.com/frommwonderland/dps/assets/dps_data.svg">
</p>
