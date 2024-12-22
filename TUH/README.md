# Epilepsy Classification Project
Author: Nguyen Tuan Khai

Graduate Project 

## Introduction 
Epilepsy, a neurological disorder causing seizures, can be debilitating. Current detection methods have limitations. This project explores using machine learning to analyze EEG signals for early seizure warnings, potentially improving patient safety and outcomes.

## Statistic
<table>
<tr><th>Length</th><th>Frequency</th></tr>
<tr><td>

| Duration | Epilepsy | No epilepsy |
| :--- | :---------- |:---------- |
| under 1m | 126 | 123 |
| From 1 to 10m | 667 | 194 |
| Between 10 and 30m | 905 | 173 |
| More than 30m | 87 | 23 |
| **Total** | **1785** | **513** |

</td><td>

| Frequency | Epilepsy | No Epilepsy |
| :--- | :---------- |:---------- |
| 250.0 Hz | 386 | 295 |
| 256.0 Hz | 1144 | 210 |
| 1000.0 Hz | 17 | 3 |
| 512.0 Hz | 168 | 0 |
| 400.0 Hz | 70 | 5 |
| **Total** | **1785** | **513** |

</td></tr> </table>


## How to run
Create virtual environment with python

```bash
python3 -m venv venv
```

Install required package

```bash
pip install -r requirements.txt
```

## GUI
Download model through the link in Saved-model/DL-model/model.txt and move in to the UI to run the code
```bash
streamlit run homepage.py
```

