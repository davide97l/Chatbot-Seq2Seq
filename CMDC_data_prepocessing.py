from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import re
import os
import codecs
from io import open
import argparse


def printLines(file, n=10):
    """Open a file and read the first n lines"""
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def loadLines(fileName, fields):
    """Splits each line of the file into a dictionary of fields
    The inner dictionary has the following keys: "lineID", "characterID", "movieID", "character", "text"
    The outer dictionary has 1 key: "lineID" corresponding to the "lineID" key of its internal object"""
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")  # separator
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


def extractSentencePairs(conversations):
    """Extracts pairs of sentences from conversations"""
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def loadConversations(fileName, lines, fields):
    """Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*"""
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")  # delimiter
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list convObj["utteranceIDs"] == "['L194', 'L195', 'L196', 'L197']\n"
            # to convObj["utteranceIDs"] == ['L194', 'L195', 'L196', 'L197']
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            # ConvObj contains all the lines with the same lineID (the ones which belong to the same conversation)
            conversations.append(convObj)
            # conversation is a three-dictionaries nested object
    return conversations


def make_formatted_file(filename="formatted_movie_lines.txt", corpus="cornell movie-dialogs corpus"):
    """Create a file containing all pairs of sentences
    Args:
    - filename: name of the file containing the formatted lines
    - corpus: directory containing the files representing the corpus
     Return:
    - location of the new file created
    """
    # Define path to new file
    datafile = os.path.join(corpus, filename)
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize field ids
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)
    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    return datafile


if __name__ == '__main__':
    """
        python CMDC_data_preprocessing.py -n "formatted_movie_lines.txt" -cl "data/cornell movie-dialogs corpus"
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--formatted_txt_name", type=str, required=True,
                    help="name of the output formatted file (txt format)")
    ap.add_argument("-cl", "--corpus_location", type=str, required=True,
                    help="Directory where the input corpus is located")
    args = vars(ap.parse_args())

    corpus_location = args["corpus_location"]
    new_filename = args["formatted_txt_name"]

    df = make_formatted_file(new_filename, corpus_location)
    print("Corput formatted")
    printLines(df)
