from flask import Flask, render_template, request
from generate_summaries import generate_summaries
from example_texts import *

app = Flask(__name__)

# length
# mit algos genauer beschäftigen!!

# waiting circle

# popup explanation
# explanation of algos

@app.route('/')
@app.route('/index.html', methods=['GET', 'POST'])
def index():

    if request.method == "GET":
        data = {
            "box1": True,
            "extractSliderValue": 3
        }
        return render_template('index.html', data=data)

    if request.method == "POST":

        data_post = request.form

        check_ls = False
        if data_post.get("selectLS") == "on":
            check_ls = True
        check_lsa = False
        if data_post.get("selectLSA") == "on":
            check_lsa = True
        check_kl = False
        if data_post.get("selectKL") == "on":
            check_kl = True
        check_bsum = False
        if data_post.get("selectBSUM") == "on":
            check_bsum = True
        check_wtf = False
        if data_post.get("selectTF") == "on":
            check_wtf = True
        check_tfidf = False
        if data_post.get("selectWTF") == "on":
            check_tfidf = True
        check_tr = False
        if data_post.get("selectTR") == "on":
            check_tr = True
        check_fm = False
        if data_post.get("selectFM") == "on":
            check_fm = True

        check_t5 = False
        model_t5 = ""
        if data_post.get("selectT5") == "on":
            check_t5 = True
            model_t5 = data_post.get("selectT5Model")
        check_bart = False
        model_bart = ""
        if data_post.get("selectBART") == "on":
            check_bart = True
            model_bart = data_post.get("selectBartModel")
        check_pegasus = False
        model_pegasus = ""
        if data_post.get("selectPEGASUS") == "on":
            check_pegasus = True
            model_pegasus = data_post.get("selectPegasusModel")

        summary_length_extract = data_post.get("extractSlider")

        if data_post["submit"] == "example":
            example_text_to_show = example_text_1

            data = {
                "box1": check_ls,
                "box2": check_lsa,
                "box3": check_kl,
                "box4": check_bsum,
                "box5": check_wtf,
                "box6": check_tfidf,
                "box7": check_tr,
                'box8': check_fm,
                "box9": check_t5,
                "box10": check_bart,
                "box11": check_pegasus,
                "selectedT5": model_t5,
                "selectedBART": model_bart,
                "selectedPegasus": model_pegasus,
                "inputText": example_text_to_show,
                "extractSliderValue": summary_length_extract
            }

            return render_template('index.html', data=data)

        if data_post["submit"] == "summarize":

            selected_algorithms = [check_ls, check_lsa, check_kl, check_bsum, check_wtf, check_tfidf, check_tr, check_fm,
                                   check_t5, model_t5, check_bart, model_bart, check_pegasus, model_pegasus]

            input_text = data_post.get("inputTextArea")

            summary_dict = generate_summaries(input_text, selected_algorithms, summary_length_extract)

            data = {
                "box1": check_ls,
                "box2": check_lsa,
                "box3": check_kl,
                "box4": check_bsum,
                "box5": check_wtf,
                "box6": check_tfidf,
                "box7": check_tr,
                'box8': check_fm,
                "box9": check_t5,
                "box10": check_bart,
                "box11": check_pegasus,
                "selectedT5": model_t5,
                "selectedBART": model_bart,
                "selectedPegasus": model_pegasus,
                "inputText": input_text,
                "extractSliderValue": summary_length_extract
            }
            return render_template('index.html', data=data, summaryResult=summary_dict)

        if data_post["submit"] == "delete":
            data = {
                "box1": True,
                "box2": False,
                "box3": False,
                "box4": False,
                "box5": False,
                "box6": False,
                "box7": False,
                "box8": False,
                "box9": False,
                "box10": False,
                "box11": False,
                "selectedT5": "baseT5",
                "selectedBART": "largeCnnBart",
                "selectedPegasus": "largePegasus",
                "extractSliderValue": 3
            }

            return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
