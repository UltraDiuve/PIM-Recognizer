import os

from flask import Flask
from flask_session import Session


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SESSION_FILE_THRESHOLD=40,
        SESSION_TYPE='filesystem',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    Session(app)

    from . import portal_home
    app.register_blueprint(portal_home.bp)
    app.add_url_rule('/', endpoint='index')

    return app
