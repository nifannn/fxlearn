import uwsgi
from . import faiss_index
from flask import Blueprint, jsonify, request
import numpy as np
import faiss
from . import config
from . import util


bp = Blueprint('faiss_search', __name__)
UPDATE_INDEX_SIGNAL = 1
UPDATE_EMBED_SIGNAL = 2

def update_embed(signal=None):
    if signal:
        util.download_embedding()
        uwsgi.signal(UPDATE_INDEX_SIGNAL)

def update_index(signal=None):
    if signal:
        set_faiss_index()

def set_faiss_index():
    bp.faiss_index = faiss_index.FaissIndex(util.load_embedding())
    print("Set faiss index successfully")

@bp.record_once
def init_index(setup_state):
    set_faiss_index()
    uwsgi.register_signal(UPDATE_INDEX_SIGNAL, "workers", update_index)
    uwsgi.register_signal(UPDATE_EMBED_SIGNAL, "worker", update_embed)
    uwsgi.add_timer(UPDATE_EMBED_SIGNAL, config.update_seconds)

@bp.route("/search_by_id", methods=['POST'])
def search_by_id():
    try:
        json_data = request.get_json(force=True)
        print("Receive Json data : ", json_data)

        results = bp.faiss_index.search_by_ids(json_data['ids'], json_data['k'])
        return jsonify(results)
    except Exception as e:
        print('Server error', e)
        return 'Server error', 500

@bp.route("/search_by_vector", methods=['POST'])
def search_by_vector():
    try:
        json_data = request.get_json(force=True)
        print("Receive Json data : ", json_data)
        results = bp.faiss_index.search_by_vectors(json_data['vectors'], json_data['k'])
        return jsonify(results)
    except Exception as e:
        print('Server error', e)
        return 'Server error', 500

@bp.route("/get_goods_id", methods=['POST'])
def get_goods_id():
    try:
        json_data = request.get_json(force=True)
        print("Receive Json data : ", json_data)
        results = bp.faiss_index._ix2id[:json_data['n']]
        return jsonify(results)
    except Exception as e:
        print('Server error', e)
        return 'Server error', 500
