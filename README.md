# EN.520.439 MLMA Final Project 
# Title: Nurse Stress Detection Using Machine Learning on Multi-Modal Wearable Sensors Data
## Group: Eric Guan, Iris Kwon, Nadia Momtaz, Aditya Shrinivasan

### Motivation 
Nursing is a profession characterized by frequent exposure to high-stress environments, leading to occupational stress, burnout, and post-traumatic stress (Melnyk et al., 2020). Higher rates of occupational stress and burnout are shown to be associated with numerous adverse patient events such as worse patient safety (i.e. patient falls) and medication errors (Li et al., 2024). Measurement and reduction of caregiver stress is therefore both a patient safety and occupational issue. Improved assessment of clinician stress may inform targeted interventions to reduce burnout and clinician turnover, ultimately enhancing care delivery. Traditional methods for monitoring nurse wellbeing typically rely on retrospective self-reporting, which is frequently biased and does not capture physiological nuances of acute stress during a particular shift. Wearable devices allow for continuous and unobtrusive collection of markers correlated with stress, as supported by prior literature using heart rate variability (HRV) and electrodermal activity (EDA) features to detect acute stress (Dalmeida & Masala, 2021; Almadhor et al., 2023). We will develop and evaluate a machine learning model that detects nurses’ stress levels using physiological data compiled from wearables. The final model will generate stress level detection from wearable data inputs. A user interface will be developed to facilitate seamless data upload and interpretation. Empowering nurses or staff coordinators with actionable insights about staff stress levels will enable healthcare systems to better allocate resources, support staff, and mitigate burnout.

### Significance 
Developing a robust system capable of detecting stress would profoundly benefit healthcare sustainability and patient safety. It would allow for proactive, targeted interventions that reduce the progression towards chronic burnout, improve the wellbeing of nurses, and indirectly reduce the risk of adverse patient outcomes through better quality of care and patient safety (Li et al., 2024; Jun et al., 2021; Hall et al., 2016). Over the long term, the ability to accurately detect stress state via wearables can provide longitudinal data to inform operational and staffing practices, yielding higher clinician retention and a more resilient healthcare workforce.

### Data Description
The Nurse Stress Prediction - Wearable Sensors dataset, originally published on Dryad (https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f#usage) by Hosseini et al. from the University of Louisiana at Lafayette, was used for this study. The pre-merged CSV file consolidating all participant signals into a single structured dataset was accessed from Kaggle (https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors/data). The merged dataset (merged_data.csv) contains approximately 11.5 million rows and 9 columns (Table 1). 

Data was collected over one-week periods from 15 female nurses aged 30-55 years working regular hospital shifts in two phases: Phase 1 (April-August, 2020) and Phase II (October-December, 2020), concurring with the COVID-19 outbreak. Participants screened for pregnancy, heavy smoking, mental disorders, chronic and/or cardiovascular diseases were excluded from the study. The study was approved by the University’s Institutional Review Board (FA19-50 INFOR), and all subject data was anonymized using unique identifiers. 

All participants wore Empatica E4 wearable wristbands for the continuous recording of physiological signals that focused primarily on Galvanic Skin Response and Blood Volume Pulse (BVP) measurements. The wristband tracked measurements including heart rate (HR), skin temperature, and electrodermal activity (EDA). Participants self-reported stress levels via periodic smartphone-administered surveys. The dataset source does not specify exactly when (time/date) the participants were required to report their stress levels. 

- High-frequency physiological signals: 
    - Heart Rate (HR): beats per minute (6,268 unique values)
    - Electrodermal Activity (EDA): skin conductance (274,452 unique values)
    - Skin Temperature (TEMP): skin temperature in celsius (599 unique values)
    - Accelerometer (X, Y, Z): three-axis accelerometer/orientation data (256 uniqe values each) 
    - Datetime: timestamp of each reading, including year, day, and exact time 
    - Label: self-reported stress level (0=no stress, 1=medium, 2=high)

Each nurse data spans from a minumum of 1 to maximum of 13 days and consists of time-stamped observations. 
