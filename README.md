<h1>CS412-IML Young People Empathy Predictor</h1>
<h4>Dataset source: </h4> 
  <p> The data set used for this project is available at https://www.kaggle.com/miroslavsabo/young-people-survey/
  </p>
  <hr>
<h4>Setup: </h4> 
  <p> After cloning the repository,
  <pre>
  1. cd CS412-IML/
  2. Make sure you have python3 installed
  3. Create a virtual environment by running,
            virtualenv -p python3 venv
  4. Run, source venv/bin/activate
  5. Next install the required packages using,
            pip install -r requirements.txt </pre>
<strong>Note:</strong> To avoid errors, please make sure you have python3-tk and graphviz installed in your system. If you don't have them installed, run the below commands (for debian),
  <pre>
  sudo apt-get install python3-tk
  sudo apt-get install graphviz  </pre>
  </p>
  <hr>
<h4>How to run: </h4> 
  <p>  
  This program takes 2 values (train/test) for parameter s (step) , 4 values of (dmc/svc/dtc/rfc) for parameter m (model) and 
  2 values (yes/no) for parameter p (data preparation).
  <pre>
  dmc: Dummy Classifier
  dtc: Decision Tree Classifier
  svc: Supoort Vector Classifier
  rfc: Random Forest Classifier </pre>
  
  <h5>Testing</h5>
  To run testing with Random Forest model
  <pre>python main.py -s test -m rfc </pre>

  <h5>Training without data preparation</h5>
  To run training with Random Forest model skipping data preparation steps
  <pre>python main.py -s train -m rfc </pre>
  
  <h5>Training with data preparation</h5>
  To run training with Random Forest model after data preparation steps(preprocessing, feature selection, etc.). Also, make sure the raw data file responses.csv is present in raw_data/ folder
  <pre>python main.py -s train -m rfc -p yes </pre>



<h5>Other options:</h5>

<pre>python main.py --help</pre>
<pre>
required arguments:
  -s {train,test}, --step {train,test}
                        train/test step
  -m {dmc,svc,dtc,rfc}, --model {dmc,svc,dtc,rfc}
                        model to use: dmc=DummyClassifer, svc=SVCClassifier,
                        dtc=DecisionTreeClassifier, rfc=FandomForestClassifier

optional arguments:
  -h, --help            show this help message and exit
  -p {yes,no}, --prep {yes,no}
                        do data prep steps: preprocessing, splitting, feature
                        selection (ignored when used with test step option)  </pre>
  </p>


