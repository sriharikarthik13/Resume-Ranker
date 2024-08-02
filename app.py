import os
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField,FileField,StringField
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Flask, render_template, request
from sqlalchemy import LargeBinary 
import fitz  # PyMuPDF
from io import BytesIO
from rresume import ranking


def extract_text_from_pdf(pdf_path):
   doc = fitz.open(pdf_path)
   text = ""
   for page_num in range(doc.page_count):
       page = doc.load_page(page_num)
       text += page.get_text()
   return text


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/uploads"
app.config['SECRET_KEY']='secret'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "data.sqlite")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
Migrate(app,db)

class resume_db(db.Model):
    __tablename__="resumedb"
    id = db.Column(db.Integer,primary_key=True,auto_increment=True)
    filename = db.Column(db.String)
    resume_file = db.Column(LargeBinary)

class jobdescription_db(db.Model):
    __tablename__="jddb"
    id = db.Column(db.Integer,primary_key=True,auto_increment=True)
    jd_id = db.Column(db.Integer)
    job_description = db.Column(db.String)


class jdform(FlaskForm):
    jobid = IntegerField("Job Description ID")
    jdescription = StringField("Job Description")
    submit = SubmitField("SUBMIT")

class resumeform(FlaskForm):
    resume = FileField("Resume")
    submit = SubmitField("SUBMIT")

class formforresume(FlaskForm):
    resumefile = FileField("Resume")
    submit = SubmitField("SUBMIT")

class rankform(FlaskForm):
    jdesc_id = IntegerField("Job Description ID")
    submit = SubmitField("SUBMIT")

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('home.html')


@app.route('/jd',methods=['GET','POST'])
def jd_index():
    form = jdform()
    if form.validate_on_submit():
        jobidval = form.jobid.data
        jdval = form.jdescription.data
        new_jd = jobdescription_db(jd_id=jobidval , job_description=jdval)
        db.session.add(new_jd)
        db.session.commit()
        return render_template("success.html")
    return render_template('form.html',form = form)

@app.route('/resume',methods=['GET','POST'])
def resume_index():
    form = formforresume()
    if form.validate_on_submit():
        print("ss")
        resume_val = form.resumefile.data
        print(resume_val)
        file_name = form.resumefile.data.filename
        data = resume_val.read()
        print(resume_val)
        new_resume = resume_db(filename=file_name,resume_file = data)
        db.session.add(new_resume)
        db.session.commit()
        return render_template("success.html")

    return render_template('form_r.html',form = form)

@app.route('/read',methods=['GET','POST'])
def read():
    form = rankform()
    if form.validate_on_submit():
        jdval = form.jdesc_id.data

        job_description = jobdescription_db.query.filter_by(jd_id=jdval).first().job_description
        

    
        resume = resume_db.query.all()
        lista=[]
        for i in resume:
            lista.append(i.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], i.filename)
            with open(file_path,'wb') as f:
                f.write(i.resume_file)
        listscore = []

        for i in lista:
            score = round(ranking(job_description,f'./static/uploads/{i}'),2)
            listscore.append(score)
        
        print(lista)
        print(listscore)
        for i in range(0,len(listscore)):
            for j in range(0,len(listscore)-i-1):
                if listscore[j] < listscore[j+1]:
                    listscore[j],listscore[j+1] = listscore[j+1],listscore[j]
                    lista[j],lista[j+1] = lista[j+1],lista[j]
        print(lista)
        print(listscore)
        lenv = len(listscore)
        resume_details = list(zip(range(1,lenv+1),lista,listscore))

        return render_template("result.html",resume_details= resume_details)
    


    return render_template("rankform.html",form=form)


@app.errorhandler(500)
def errorpage(error):
    """Flask View for the page to display the error message"""
    return render_template("error.html") ,500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("3000"))


