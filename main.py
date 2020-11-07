from flask import Flask ,render_template,request
app = Flask(__name__)
import pickle

@app.route("/",methods=["GET","POST"])
def hello():
    if request.method == "POST":
        mydict=request.form
        fever=int(mydict["fever"])
        age=int(mydict["age"])
        bodypain=int(mydict["bodypain"])
        runnynose=int(mydict["runnynose"])
        diffbreathing=int(mydict["diffbreathing"])
        print(fever)
        print(age)
        print(bodypain)
        print(runnynose)
        print(diffbreathing)
        print(mydict)
        infile = open('model.pk1','rb')
        clf2 = pickle.load(infile)
        inf_prob=clf2.predict_proba([[fever,bodypain,age,runnynose,diffbreathing]])[0][1]
        print(inf_prob)
        infile.close()
        return render_template('show.htm',inf=inf_prob)
    return render_template('index.htm')


if __name__ == '__main__':
    app.run(debug=True)
    

    