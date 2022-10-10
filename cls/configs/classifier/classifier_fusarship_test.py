_base_ = [
    './classifier_test.py',
    '../_base_/datasets/fusarship.py',
]

model = dict(classifier=dict(dim_out=3))
