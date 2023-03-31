# Readme

Dataset used : [UD_English-Atis](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4923) 

## File structure
* 2021114010_NLP_ASS2
  * **.vector_cache** 
  * **UD_English-Atis** contains English dataset
  * **idx2tag.pickle** -> saved dictionaries  
  * **tag2idx.pickle** -> saved dictionaries 
  * **word2idx.pickle** -> saved dictionaries 
  * **Report.pdf** 
  * **pos-tagger_trainer.py** -> for training the data 
  * **model.pt** -> pre-trained model
  * **pos-tagger.py** -> for testing , tagging a input sentence 


## To run the script 
### To Train the model  
```bash
python3 pos-tagger_trainer.py
```
### To run pos tagger 
```bash
python3 pos-tagger.py
```