To execute this project kindly follow the code execution order or pipeline
1, run test.py file which extracts the reddit json files from differnet subreddits(make sure to update the directory structure)
2, Now run pipline.py file which will do first level of extraction and store the json in extracted folder
3, Next execute the entity.py this will take long tiem to process
4, Now llm_chroma.py to process the jsons to chroma
5, Now chatbot.py this need to be executed last as this will open up a web page in streamlit. to run use command streamlit run chatbot.py 
