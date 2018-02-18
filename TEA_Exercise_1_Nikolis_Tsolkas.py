from nltk import sent_tokenize, word_tokenize, defaultdict
from collections import Counter
from nltk.util import ngrams
from random import randint
from operator import add
import string
import codecs
import math

# Code commented out was used at some stage (e.g. validation) but is not needed for the final run

UNKNOWN = "*UNK*"
BIGRAM_START = " *start* "  # start of sentence for bigram model
TRIGRAM_START = " *start1* *start2* "  # start of sentence for trigram model
END = " *end* "  # end of sentence
most_probable_word = ""


def isPunctuation(word):
    return all(i in string.punctuation for i in word)


class LanguageModel:

    # n : can be 2(bigram) or 3(trigram)
    def __init__(self, text, n, vocab):
        self.n = n
        self.vocabulary = vocab

        # Split sentences
        self.sentences = sent_tokenize(text)

        # Add *start* - *end* to each sentence
        for i, s in enumerate(self.sentences):
            if (self.n == 2):
                self.sentences[i] = (BIGRAM_START + s + END).lower()
            if (self.n == 3):
                self.sentences[i] = (TRIGRAM_START + s + END).lower()

        model_grams = list()

        # Create model without P(start|...)
        for s in self.sentences:
            tokens = word_tokenize(s)

            # Replace unknown words with *UNK*
            for i, w in enumerate(tokens):
                if (w not in self.vocabulary):
                    tokens[i] = UNKNOWN

            # Add the ngrams to a list
            model_grams += [gram for gram in ngrams(tokens, self.n)]

        # Build model
        # Create a dictionary that returns a dictionary
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

        # Initialize the dictionaries
        if (self.n == 2):
            for w1, w2 in model_grams:
                self.model[w1][w2] += 1
        elif (self.n == 3):
            for w1, w2, w3 in model_grams:
                self.model[(w1, w2)][w3] += 1

    def get_probability(self, sequence, break_sentences=False):
        # print(sequence)

        if(break_sentences):
            sentences = sent_tokenize(sequence)
        else:
            sentences = [sequence]

        log_prob = 0
        all_probs = list()  # used by the linear model to calculate its probabilities
        length = 0

        for s in sentences:
            # Add *start* - *end* to sentence
            if (self.n == 2):
                s = (BIGRAM_START + s + END).lower()
            if (self.n == 3):
                s = (TRIGRAM_START + s + END).lower()

            # Tokenize - list of words
            sent_tokens = word_tokenize(s)

            # Remove punctuation
            sent_tokens = [i for i in sent_tokens if not isPunctuation(i)]
            length += len(sent_tokens) - self.n

            # Replace unknown words with *UNK*
            for i, w in enumerate(sent_tokens):
                if (w not in self.vocabulary):
                    sent_tokens[i] = UNKNOWN

            sentence_grams = [gram for gram in ngrams(sent_tokens, self.n)]

            if (self.n == 2):
                for w1, w2 in sentence_grams:
                    total_count = float(sum(self.model[w1].values()))  # count of the word
                    prob = (self.model[w1][w2] + 1) / (total_count + len(vocabulary))
                    all_probs.append(prob)
                    log_prob += math.log2(prob)
            elif (self.n == 3):
                for w1, w2, w3 in sentence_grams:
                    total_count = float(sum(self.model[(w1, w2)].values()))  # count of the tuple
                    prob = (self.model[(w1, w2)][w3] + 1) / (total_count + len(vocabulary))
                    all_probs.append(prob)
                    log_prob += math.log2(prob)

        return log_prob, length, all_probs

    def predict_next_word(self, sequence):
        # print(sentence)

        # Add *start* to each sentence
        if (self.n == 2):
            sequence = (BIGRAM_START + sequence + " ").lower()
        if (self.n == 3):
            sequence = (TRIGRAM_START + sequence + " ").lower()

        # Tokenize - list of words
        sent_tokens = word_tokenize(sequence)

        # Remove punctuation
        sent_tokens = [i for i in sent_tokens if not isPunctuation(i)]

        # Replace unknown words with *UNK*
        for i, w in enumerate(sent_tokens):
            if (w not in self.vocabulary):
                sent_tokens[i] = UNKNOWN

        best_word = most_probable_word
        max_freq = 0

        if (self.n == 2):
            w1 = sent_tokens[-1]
            words = self.model[w1].items()
        elif (self.n == 3):
            w1 = sent_tokens[-2]
            w2 = sent_tokens[-1]
            words = self.model[(w1, w2)].items()

        for word, freq in words:
            if (word == UNKNOWN or word == "*end*"):
                continue

            if (freq > max_freq):
                best_word = word
                max_freq = freq

        return best_word, words


class LinearModel:

    def __init__(self, bigr_model, trigr_model, l_bigram):
        self.bigr_model = bigr_model
        self.trigr_model = trigr_model
        self.l_bigram = l_bigram
        self.l_trigram = 1-l_bigram

    def get_probability(self, sequence, break_sentences=False):
        # print(sequence)

        # Get probabilities from both models
        _, _, all_probs_bigram = self.bigr_model.get_probability(sequence, break_sentences)
        _, _, all_probs_trigram = self.trigr_model.get_probability(sequence, break_sentences)

        # Multiply each one by the corresponding lambda
        all_probs_bigram = [self.l_bigram * i for i in all_probs_bigram]
        all_probs_trigram = [self.l_trigram * i for i in all_probs_trigram]

        # Add probabilities of the same word
        all_probs = map(add, all_probs_bigram, all_probs_trigram)

        # Calculate the log probabilities
        all_probs = [math.log2(i) for i in all_probs]
        prob = sum(all_probs)

        return prob, 0, all_probs

    def predict_next_word(self, sequence):
        # print(sentence)

        # Get all predictions with the respective counts
        _, bigram_words = self.bigr_model.predict_next_word(sequence)
        _, trigram_words = self.trigr_model.predict_next_word(sequence)

        words_bigr = {}
        words_trigr = {}

        # Multiply the frequencies by the respective lambdas
        for w, f in bigram_words:
            words_bigr[w] = self.l_bigram * f
        for w, f in trigram_words:
            words_trigr[w] = self.l_trigram * f

        # Calculate the final frequencies of all words
        all_words = {}
        for i in words_bigr:
            all_words[i] = words_bigr[i] + words_trigr.get(i, 0)

        # Find the best prediction
        best_word = most_probable_word
        max_freq = 0

        for word, freq in all_words.items():
            if (word == UNKNOWN):
                continue

            if (freq > max_freq):
                best_word = word
                max_freq = freq

        return best_word, all_words


def built_random_sentence(sentence_length, vocab):
    result_sentence = ""

    for _ in range(sentence_length):
        r_int = randint(0, len(vocab) - 1)
        for word in vocab.keys():
            r_int -= 1
            if (r_int == -1):
                result_sentence += " " + word
                break
    return result_sentence.strip()


######################################################################
# (i)

# Load training data
data_path = 'D:/Data/'
with codecs.open(data_path + "train.txt", encoding='utf-8') as f:
    train_text = f.read()
print("Loaded training data")

train_tokens = word_tokenize(train_text)
print("Tokenized training data")

# Load validation data
# with codecs.open(data_path + "validate.txt", encoding='utf-8') as f:
#     validate_text = f.read()
# validation_sentences = sent_tokenize(validate_text)
# validation_tokens = word_tokenize(validate_text)
# print("Validation Words: ", len(validation_tokens))
# print("Loaded validation data")

# Load test data
with codecs.open(data_path + "test.txt", encoding='utf-8') as f:
    test_text = f.read()
test_sentences = sent_tokenize(test_text)
test_tokens = word_tokenize(test_text)
print("Test Words: ", len(test_tokens))
print("Loaded test data")

# Remove punctuation
train_tokens = [i for i in train_tokens if not isPunctuation(i)]
print("Removed punctuation")

# Count train words
counter = Counter(train_tokens)
print("Initial Words: ", len(train_tokens))
print("Initial distinct Words: ", len(counter))

# Replace rare words with "*UNK*"
for i, w in enumerate(train_tokens):
    if (counter[w] < 10):
        train_tokens[i] = UNKNOWN

# Get the vocabulary
vocabulary = Counter(train_tokens)
most_probable_word, _=vocabulary.most_common(1)[0]
print("Distinct Words: ", len(vocabulary))

# Create models
bigram_model = LanguageModel(train_text, 2, vocabulary)
trigram_model = LanguageModel(train_text, 3, vocabulary)
# linear_model1 = LinearModel(bigram_model, trigram_model, 0.1)
# linear_model2 = LinearModel(bigram_model, trigram_model, 0.2)
# linear_model3 = LinearModel(bigram_model, trigram_model, 0.3)
# linear_model4 = LinearModel(bigram_model, trigram_model, 0.4)
# linear_model5 = LinearModel(bigram_model, trigram_model, 0.5)
# linear_model6 = LinearModel(bigram_model, trigram_model, 0.6)
# linear_model7 = LinearModel(bigram_model, trigram_model, 0.7)
# linear_model8 = LinearModel(bigram_model, trigram_model, 0.8)
linear_model9 = LinearModel(bigram_model, trigram_model, 0.9)
print("Created models")

######################################################################
# (ii)

for i in range(10):
    print("")
    print("")

    # Get random test sentence
    r = randint(0, len(test_sentences) - 1)
    sentence = test_sentences[r]
    print("correct sentence : "+sentence)

    # Print log probabilities
    log_prob_bigr1, len1, _ = bigram_model.get_probability(sentence)
    rand_sentence=built_random_sentence(len1,vocabulary)
    print("random sentence : " + rand_sentence)
    print("log probability bigr (correct): " + str(log_prob_bigr1))
    log_prob_bigr2, _, _ = bigram_model.get_probability(rand_sentence)
    print("log probability bigr (random): " + str(log_prob_bigr2))

    log_prob_trigr1, len2, _ = trigram_model.get_probability(sentence)
    print("log probability trigr (correct): " + str(log_prob_trigr1))
    log_prob_trigr2, _, _ = trigram_model.get_probability(rand_sentence)
    print("log probability trigr (random): " + str(log_prob_trigr2))

    log_prob_linear91, _, _ = linear_model9.get_probability(sentence)
    print("log probability linear (correct): " + str(log_prob_linear91))
    log_prob_linear92, _, _ = linear_model9.get_probability(rand_sentence)
    print("log probability linear (random): " + str(log_prob_linear92))

####################################################################
# (iii)

# Get 10 sequences from user and predict the next word
for _ in range(10):
    sentence = input("Insert sentence : ")
    print(sentence)
    print(bigram_model.predict_next_word(sentence)[0])
    print(trigram_model.predict_next_word(sentence)[0])
    print(linear_model9.predict_next_word(sentence)[0])

####################################################################
# (iv)

# Get log probabilities

# Validate data
# bigr_prob, len1, _ = bigram_model.get_probability(validate_text)
# trigr_prob, len2, _ = trigram_model.get_probability(validate_text)
# linear_prob1, _, _ = linear_model1.get_probability(validate_text)
# linear_prob2, _, _ = linear_model2.get_probability(validate_text)
# linear_prob3, _, _ = linear_model3.get_probability(validate_text)
# linear_prob4, _, _ = linear_model4.get_probability(validate_text)
# linear_prob5, _, _ = linear_model5.get_probability(validate_text)
# linear_prob6, _, _ = linear_model6.get_probability(validate_text)
# linear_prob7, _, _ = linear_model7.get_probability(validate_text)
# linear_prob8, _, _ = linear_model8.get_probability(validate_text)
# linear_prob9, _, _ = linear_model9.get_probability(validate_text)

# Test data
bigr_prob, len1, _ = bigram_model.get_probability(test_text, True)
trigr_prob, len2, _ = trigram_model.get_probability(test_text, True)
linear_prob9, _, _ = linear_model9.get_probability(test_text, True)

# Calculate cross entropy and perplexity of models
cross_entropy_bigr = (-1)*bigr_prob/len1
cross_entropy_trigr = (-1)*trigr_prob/len2
# cross_entropy_linear1 = (-1)*linear_prob1/len2
# cross_entropy_linear2 = (-1)*linear_prob2/len2
# cross_entropy_linear3 = (-1)*linear_prob3/len2
# cross_entropy_linear4 = (-1)*linear_prob4/len2
# cross_entropy_linear5 = (-1)*linear_prob5/len2
# cross_entropy_linear6 = (-1)*linear_prob6/len2
# cross_entropy_linear7 = (-1)*linear_prob7/len2
# cross_entropy_linear8 = (-1)*linear_prob8/len2
cross_entropy_linear9 = (-1)*linear_prob9/len2

print("cross entropy bigr : "+str(cross_entropy_bigr))
print("cross entropy trigr : "+str(cross_entropy_trigr))
# print("cross entropy linear1 : "+str(cross_entropy_linear1))
# print("cross entropy linear2 : "+str(cross_entropy_linear2))
# print("cross entropy linear3 : "+str(cross_entropy_linear3))
# print("cross entropy linear4 : "+str(cross_entropy_linear4))
# print("cross entropy linear5 : "+str(cross_entropy_linear5))
# print("cross entropy linear6 : "+str(cross_entropy_linear6))
# print("cross entropy linear7 : "+str(cross_entropy_linear7))
# print("cross entropy linear8 : "+str(cross_entropy_linear8))
print("cross entropy linear9 : "+str(cross_entropy_linear9))

perplexity_bigram = 2**cross_entropy_bigr
perplexity_trigram = 2**cross_entropy_trigr
# perplexity_linear1 = 2**cross_entropy_linear1
# perplexity_linear2 = 2**cross_entropy_linear2
# perplexity_linear3 = 2**cross_entropy_linear3
# perplexity_linear4 = 2**cross_entropy_linear4
# perplexity_linear5 = 2**cross_entropy_linear5
# perplexity_linear6 = 2**cross_entropy_linear6
# perplexity_linear7 = 2**cross_entropy_linear7
# perplexity_linear8 = 2**cross_entropy_linear8
perplexity_linear9 = 2**cross_entropy_linear9

print("perplexity bigr: "+str(perplexity_bigram))
print("perplexity trigr : "+str(perplexity_trigram))
# print("perplexity linear1 : "+str(perplexity_linear1))
# print("perplexity linear2 : "+str(perplexity_linear2))
# print("perplexity linear3 : "+str(perplexity_linear3))
# print("perplexity linear4 : "+str(perplexity_linear4))
# print("perplexity linear5 : "+str(perplexity_linear5))
# print("perplexity linear6 : "+str(perplexity_linear6))
# print("perplexity linear7 : "+str(perplexity_linear7))
# print("perplexity linear8 : "+str(perplexity_linear8))
print("perplexity linear9 : "+str(perplexity_linear9))
