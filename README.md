# TriviaQuestion2NQ_Transform
## Setup & Data Download
The four .json files: **`qanta.train.json`**, **`qanta_id_to_the_answer_type_most_freq_phrase_based_on_page_dict.json`**,
**`qb_train_with_contexts_lower_nopunc_debug_Feb24.json`**, **`word_transform_dict.json`**, are all required to run the transformation.
Please make sure they are successfully cloned.

Cloning these files require git-lfs. Install git-lfs by

```
sudo apt-get install git-lfs
git lfs install
```

Then clone the datase

```
git clone https://github.com/saptab/TriviaQuestion2NQ_Transform.git
cd ./TriviaQuestion2NQ_Transform/TriviaQuestion2NQ_Transform_Dataset
```

Prior to running our code, please install all the required packages via

`pip install -r "requirements.txt"`

## Prerequisite Installation
It is required to load neuralcoref via

```
cd ..
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
cd ..
```
To get spacy library working, run the following command
```
python -m spacy download en_core_web_sm
```
## Form_Answer_Type_Frequency_Table.py
Running either **`Form_Answer_Type_Frequency_Table.py`** or **`Form_Answer_Type_Frequency_Table.ipynb`** will generate a dictionary containing qanta training questions with answer types, saved as a json file, **`qanta_train_with_answer_type_v1.json`**

There is no need to regenerate the dictionary as we are already providing the dictionary file.

## Form_Answer_Type_Frequency_Table.py
Running either **`QB_NQ_transformation_code_base.py`** or **`QB_NQ_transformation_code_base.ipynb`** will transform qb questions to a list of nq_like questions. There are several arguments that can be passed in.

`--limit` set the number of qb questions that are transformed, default set to `-1`.

`--qb_path` set the path to the qb question dataset to be transformed. Default set to our qb training dataset. You can change this argument to apply the transformation function on any qb dataset you'd like.

`--save_result` saves the nq_like result with corresponding context after transformation into a json file, default set to `False`.

`--save_only_NQlike_questions` saves only the nq_like result into a json file, default set to `False`.

'--answer_type_classifier' retrains the answer type classifier from scratch, default set to 'False' as we already provide the checkpoints.

There is no need to rerun the answer detection classifier as we are already providing the answer detection classifier checkpoints.
~                                                                               
