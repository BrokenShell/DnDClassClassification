""" D&D Class Classification
API Viewer

"""
from flask import Flask, jsonify
from classifier.controller import Classifier

__all__ = ('API',)
API = Flask(__name__)
API.arch_classifier = Classifier('classifier/class_data.csv')


@API.route('/')
def home():
    return jsonify(Class=API.arch_classifier.random_class())


@API.route('/<user_input>')
def search(user_input):
    return jsonify(Class=API.arch_classifier(user_input))


if __name__ == '__main__':
    API.run(debug=True)
