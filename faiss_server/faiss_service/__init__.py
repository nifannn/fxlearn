from flask import Flask
from . import faiss_search

def create_app():
    app = Flask(__name__)

    app.register_blueprint(faiss_search.bp)

    @app.route("/ping")
    def ping():
        return "pong"

    return app