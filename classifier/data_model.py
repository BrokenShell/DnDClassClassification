""" D&D Class Classification
Data Model

"""
import csv

__all__ = ('DataModel', 'dict_to_string')


class DataModel:
    __slots__ = ('filename', 'class_details', 'training_data')

    def __init__(self, filename):
        self.filename = filename
        self.class_details = []

        with open(self.filename) as f:
            doc = csv.reader(f)
            keys = next(doc)

            for row in doc:
                self.class_details.append({k: v for k, v in zip(keys, row)})

        self.training_data = ['. '.join(char.values()) for char in self.class_details]


def dict_to_string(obj: dict) -> str:
    """ Returns the string representation of a dictionary """
    return '\n'.join(f'{k}: {v}' for k, v in obj.items())


if __name__ == '__main__':
    data = DataModel('classifier/class_data.csv')
    print(data.training_data[0])
