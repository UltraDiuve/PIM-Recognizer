from flask import (
    Blueprint, render_template, request, redirect, url_for, session,  
    make_response,  # flash
)
# from werkzeug.exceptions import abort
import joblib
from pathlib import Path
from src.pimpdf import PDFDecoder
import re
import pandas as pd
import os


bp = Blueprint('home', __name__)
simselector = joblib.load(Path('.') / 'portal' / 'similarity_selector.pkl')


def splitter(text):
    regex = r'\s*\n\s*\n\s*'
    return(re.split(regex, text))


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        session.clear()
        attached = request.files['attached']
        text = PDFDecoder.content_to_text(attached)
        splitted = pd.Series([splitter(text)])
        result = simselector.predict(splitted).iloc[0]
        session['result'] = result
        attached.stream.seek(0)
        session['filecontent'] = attached.read()
        attached.stream.seek(0)
        attached.save(os.path.join('./portal/uploads/', 'FT.pdf'))
        return redirect(url_for('home.result'))
    else:
        return render_template('home/index.html')


@bp.route('/result', methods=('GET', ))
def result(result=None, filecontent=None):
    result = session['result']
    filecontent = session['filecontent']
    return(render_template('result/result.html',
                           result=result,
                           filecontent=filecontent,
                           ))

# TODO : remove, not needed anymore since not using PDF.js
@bp.route('/pdfviewer', methods=('GET', ))
def viewer():
    return(render_template('web/viewer.html'))


# TODO : finir ici. Ca remonte un contenu que le navigateur n'interprète pas.
@bp.route('/currentfile.pdf', methods=('GET', ))
def currentfile():
    response = make_response(session['filecontent'])
    response.headers['Content-Type'] = 'application/pdf'
    return(response)
