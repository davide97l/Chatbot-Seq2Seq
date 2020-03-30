# Chatbot-Seq2Seq

Simple chatbot implemented with a GRU recurrent sequence-to-sequence model with attention using Pytorch.

## How to use it

1) **Prepare your dataset**
```
  sentence1A\tsententence1B\n
  sentence2A\tsententence2B\n
  how are you?\tI'fine\n
  ...
```
Where `sentenceNA` is the input sequence ad `sentenceNB` is the target sentence. Make sure each line contains one pair and that each pair is sepated with a tab (\t).
If you want to use the predefined dataset run the command:
```
  python CMDC_data_preprocessing.py -n "formatted_movie_lines.txt" -cl "data/cornell movie-dialogs corpus"
```
`-n` in the name of the new formatted dataset and `cl` is the location of the original corpus.

2) **Train your model**

Run the following command to train and evaluate your model and save it in a new `my_model` directory.
```
  python chatbot.py -mn "cornell_movie_dialogs_chatbot" -cl "data/cornell movie-dialogs corpus/formatted_movie_lines.txt -sd "my_model" -tr -ev
```
Many more options are available such as setting the input size, learning rate, attention method, load a previous trained model, ecc. You can use the command `--help` to visualize them.
