from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from summarizer import Summarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from algorithms import *


def generate_summaries(input_text, selected_algorithms, amount_sentences_in_summary):

    summary_dict = {}

    parser_luhn_lsa_kl = PlaintextParser.from_string(input_text, Tokenizer("english"))

    # Luhn Summarizer
    if selected_algorithms[0]:
        luhn_summarizer = LuhnSummarizer()
        luhn_summary = luhn_summarizer(parser_luhn_lsa_kl.document, amount_sentences_in_summary)
        luhn_summary = array_transfomer(luhn_summary)
        summary_dict["Luhn Summarizer"] = luhn_summary

    # LSA Summarizer
    if selected_algorithms[1]:
        lsa_summarizer = LsaSummarizer()
        lsa_summary = lsa_summarizer(parser_luhn_lsa_kl.document, amount_sentences_in_summary)
        lsa_summary = array_transfomer(lsa_summary)
        summary_dict["LSA Summarizer"] = lsa_summary

    # KL-Sum Summarizer
    if selected_algorithms[2]:
        kl_summarizer = KLSummarizer()
        kl_summary = kl_summarizer(parser_luhn_lsa_kl.document, amount_sentences_in_summary)
        kl_summary = array_transfomer(kl_summary)
        summary_dict["KL-Sum Summarizer"] = kl_summary

    # BERT Summarizer
    if selected_algorithms[3]:
        model = Summarizer()
        sum_length = 0
        if int(amount_sentences_in_summary) < 5:
            sum_length = int(amount_sentences_in_summary)-1
        else:
            sum_length = int(amount_sentences_in_summary)
        bsum_summary = model(input_text, num_sentences=sum_length)
        bsum_summary = sent_tokenize(bsum_summary)
        bsum_summary = array_transfomer(bsum_summary)
        summary_dict["BERT Summarizer"] = bsum_summary

    # Weighted Term Frequency
    if selected_algorithms[4]:
        preprocessed_sentences = preprocessing_input_text(input_text)
        scored_sentences_wtf = weighted_term_frequency(preprocessed_sentences)
        sorted_sentences_wtf = sorted(scored_sentences_wtf, key=lambda x: -x[2][0])
        summary_wtf = generate_summary(sorted_sentences_wtf, amount_sentences_in_summary)
        summary_dict["Weighted Term Frequency"] = summary_wtf

    # TF-IDF
    if selected_algorithms[5]:
        preprocessed_sentences = preprocessing_input_text(input_text)
        scored_sentences_tf_idf = tf_idf(preprocessed_sentences)
        sorted_sentences_tf_idf = sorted(scored_sentences_tf_idf, key=lambda x: -x[2][1])
        summary_tf_idf = generate_summary(sorted_sentences_tf_idf, amount_sentences_in_summary)
        summary_dict["TF-IDF"] = summary_tf_idf

    # Textrank
    if selected_algorithms[6]:
        preprocessed_sentences = preprocessing_input_text(input_text)
        scored_sentences_textrank = textrank(preprocessed_sentences)
        sorted_sentences_textrank = sorted(scored_sentences_textrank, key=lambda x: -x[2][2])
        summary_textrank = generate_summary(sorted_sentences_textrank, amount_sentences_in_summary)
        summary_dict["Textrank"] = summary_textrank

    # Feature Mix
    if selected_algorithms[7]:
        preprocessed_sentences = preprocessing_input_text(input_text)
        scores_wtf = weighted_term_frequency(preprocessed_sentences)
        scores_tfidf = tf_idf(scores_wtf)
        scores_textrank = textrank(scores_tfidf)
        scores_length = sentence_length(scores_textrank)
        scores_position = sentence_position(scores_length)
        scores_names = ooccurences_named_entities(scores_position)
        scores_numbers = amount_numerals(scores_names)
        print(scores_numbers)
        total_score_mix = total_feature_mix_score(scores_numbers)
        sorted_sentences_mix = sorted(total_score_mix, key=lambda x: -x[2])
        summary_mix = generate_summary(sorted_sentences_mix, amount_sentences_in_summary)
        summary_dict["Feature Mix"] = summary_mix

    # T5 Transformer
    if selected_algorithms[8]:
        if str(selected_algorithms[9]) == "baseT5":
            model_t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            model_t5 = T5ForConditionalGeneration.from_pretrained('t5-large')
            tokenizer = T5Tokenizer.from_pretrained('t5-large')
        input_ids = tokenizer.encode("summarize: " + input_text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model_t5.generate(input_ids, max_length=1000, min_length=40,  length_penalty=2.0,  num_beams=4,  early_stopping=True)
        t5_summary = tokenizer.decode(summary_ids[0])
        t5_summary = t5_summary.replace("<pad>", ""). replace("</s>", "")
        t5_summary = sent_tokenize(t5_summary)
        t5_summary = [sentence.lstrip() for sentence in t5_summary]
        t5_summary = [sentence.capitalize() for sentence in t5_summary]
        summary_dict["T5 Transformer"] = t5_summary

    # BART Transformer
    if selected_algorithms[10]:
        if str(selected_algorithms[11]) == "largeCnnBart":
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        else:
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
            model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
        inputs = tokenizer.batch_encode_plus([input_text], return_tensors='pt')
        summary_ids = model_bart.generate(inputs['input_ids'], max_length=150, min_length=40, early_stopping=True)
        bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        bart_summary = sent_tokenize(bart_summary)
        summary_dict["BART Transformer"] = bart_summary

    # Pegasus Transformer
    if selected_algorithms[12]:
        if str(selected_algorithms[13]) == "largePegasus":
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
            model_pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
        else:
            tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
            model_pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
        inputs = tokenizer.batch_encode_plus([input_text], return_tensors='pt')
        summary_ids = model_pegasus.generate(inputs['input_ids'], max_length=150, min_length=40, early_stopping=True)
        pegasus_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        pegasus_summary = pegasus_summary.replace("<n>", "")
        pegasus_summary = sent_tokenize(pegasus_summary)
        summary_dict["PEGASUS Transformer"] = pegasus_summary

    print(summary_dict)
    return summary_dict
