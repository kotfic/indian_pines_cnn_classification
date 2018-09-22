from __future__ import print_function
import click
from .create import preprocess_data as _preprocess_data
from .train import train_model as _train_model
from .validate import validate as _validate
from .validate import classify as _classify



@click.group()
def main():
    pass


@main.command()
@click.argument('data_path', type=click.STRING)
@click.argument('output_path', type=click.STRING)
@click.option('--num-components', type=click.INT, default=30)
@click.option('--window-size', type=click.INT, default=5)
@click.option('--test-ratio', type=click.FLOAT, default=0.25)
def preprocess_data(data_path, output_path, num_components, window_size, test_ratio):
    path = _preprocess_data(
        data_path=data_path,
        output_path=output_path,
        numComponents=num_components,
        windowSize=window_size,
        testRatio=test_ratio)

    print(path)





@main.command()
@click.argument('data_path', type=click.STRING)
@click.argument('model_path', type=click.STRING)
@click.option('--num-components', type=click.INT, default=30)
@click.option('--window-size', type=click.INT, default=5)
@click.option('--test-ratio', type=click.FLOAT, default=0.25)
def train_model(data_path, model_path, num_components, window_size, test_ratio):
    path = _train_model(
        data_path=data_path,
        model_path=model_path,
        numPCAcomponents=num_components,
        windowSize=window_size,
        testRatio=test_ratio)

    print(path)


@main.command()
@click.argument('model_path', type=click.STRING)
@click.argument('test_data_path', type=click.STRING)
@click.option('--num-components', type=click.INT, default=30)
@click.option('--window-size', type=click.INT, default=5)
@click.option('--test-ratio', type=click.FLOAT, default=0.25)
def validate(model_path, test_data_path, num_components, window_size, test_ratio):
    loss, accuracy, classification, confusion = _validate(
        model_path=model_path,
        test_data_path=test_data_path,
        numPCAcomponents=num_components,
        windowSize=window_size,
        testRatio=test_ratio)

    print('{} Test loss (%)'.format(loss))
    print('{} Test accuracy (%)'.format(accuracy))
    print('\n')
    print('{}'.format(classification))
    print('\n')
    print('{}'.format(confusion))




@main.command()
@click.argument('model_path', type=click.STRING)
@click.argument('data_path', type=click.STRING)
@click.option('--ground-path', type=click.STRING, default='ground_truth.jpg')
@click.option('--classification-path', type=click.STRING, default='classification.jpg')
@click.option('--num-components', type=click.INT, default=30)
@click.option('--window-size', type=click.INT, default=5)
def classify(model_path, data_path, ground_path, classification_path, num_components, window_size):
    gp, cp = _classify(
        model_path=model_path,
        data_path=data_path,
        ground_path=ground_path,
        classification_path=classification_path,
        patch_size=window_size,
        numComponents=num_components
    )
    print(gp)
    print(cp)
