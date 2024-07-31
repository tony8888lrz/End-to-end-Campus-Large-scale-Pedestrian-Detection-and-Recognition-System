# End-to-end-Campus-Large-scale-Pedestrian-Detection-and-Recognition-System
Constructed an accurate detection framework with self-collected and publicly labeled datasets, attaining a 97% accuracy rate, and conducted end-to-end training and deployment of the model.  
 

## Usage
### Reproduce the system
1. Create a new conda environment:
```{py}
  conda create --name bms python=3.9
  activate bms       # Windows
  conda activate bms # Linux
```
2. Clone this repo:
```
git clone [https://github.com/tony8888lrz/SWU-BMS](https://github.com/tony8888lrz/SWU-BMS/)
cd SWU-BMS
```
3. Install required packages by typing
```
pip install -r requirements.txt
```
4. Run book_management_sys.py to reproduce the management system.
```
python book_management_sys.py runserver
