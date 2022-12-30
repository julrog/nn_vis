import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import transforms
from matplotlib.axes import Axes

from definitions import BASE_PATH
from utility.file import EvaluationFile


def setup_plot():
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('figure', titlesize=14)


def load_data(name: str, importance_name: str, timed_name: bool = False) -> Dict[Any, Any]:
    evaluation_file: EvaluationFile = EvaluationFile(name)
    evaluation_file.read_data(timed_name)
    return evaluation_file.data_cache[importance_name]


def save_plot(name: str):
    directory_path: str = os.path.join(
        BASE_PATH, os.path.join('storage', 'evaluation'))
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path: str = '%s/%s.svg' % (directory_path, name)
    plt.savefig(file_path)
    file_path = '%s/%s.jpg' % (directory_path, name)
    plt.savefig(file_path)


def create_importance_plot(filename: str, importance_name: str, timed_name: bool = False, show: bool = False):
    data: Dict[Any, Any] = load_data(filename, importance_name, timed_name)

    converted_data: List[List[Any]] = []
    for percent, percent_data in data.items():
        for importance_type, importance_type_data in percent_data.items():
            importance_type_name: str = 'BN Node \u03B3 and Edge \u03B1' if importance_type == 'BNN_EDGE' \
                else 'BN Node \u03B3' if importance_type == 'BNN_ONLY' else 'Edge \u03B1' if \
                importance_type == 'EDGE_ONLY' else 'None'
            if importance_type_name != 'None':
                plot_point: List[Any] = [importance_type_name, int(percent),
                                         float(importance_type_data['train_accuracy'])]
                converted_data.append(plot_point)

    df = pd.DataFrame(converted_data,
                      columns=['Importance Calculation Method', 'Pruned Edge Percentage', 'Prediction Accuracy'])

    df = df.pivot(index='Pruned Edge Percentage',
                  columns='Importance Calculation Method', values='Prediction Accuracy')
    plot: Axes = df.plot(legend=True)
    plot.set_ylabel('Prediction Accuracy')
    plot.set_facecolor('#F4F4F4')
    plot.grid(color='white', linestyle='-', linewidth=2)
    for line in plot.lines:
        plt.setp(line, linewidth=3)
    plot.axvline(x=90, color='red')
    trans = transforms.blended_transform_factory(
        plot.get_yticklabels()[0].get_transform(), plot.transData)
    plot.text(0.9, -0.025, '{:.0f}'.format(90), color='red', transform=trans,
              va='bottom', ha='center')

    save_plot(importance_name)

    if show:
        plt.show()
    plt.close()


def create_importance_plot_compare_regularizer(filename: str, importance_names: List[str], check_importance_type: str,
                                               timed_name: bool = False, show: bool = False):
    plt.rcParams['legend.loc'] = 'lower left'
    converted_data: List[List[Any]] = []
    for importance_name in importance_names:
        importance_label_name: str = 'L2' if '_l2' in importance_name else 'L1 + L2' if 'l1l2' in importance_name \
            else 'L1' if 'l1' in importance_name else 'Without'
        data: Dict[Any, Any] = load_data(filename, importance_name, timed_name)

        for percent, percent_data in data.items():
            for importance_type, importance_type_data in percent_data.items():
                if importance_type == check_importance_type:
                    plot_point: List[Any] = [importance_label_name, int(percent),
                                             float(importance_type_data['train_accuracy'])]
                    converted_data.append(plot_point)

    df = pd.DataFrame(converted_data,
                      columns=['Regularizer', 'Pruned Edge Percentage', 'Prediction Accuracy'])

    df = df.pivot(index='Pruned Edge Percentage',
                  columns='Regularizer', values='Prediction Accuracy')
    plot: Axes = df.plot(legend=True)
    plot.set_ylabel('Prediction Accuracy')

    plot.set_facecolor('#F4F4F4')
    plot.grid(color='white', linestyle='-', linewidth=2)
    for line in plot.lines:
        plt.setp(line, linewidth=3, alpha=0.6)
    plot.axvline(x=90, color='red')
    trans = transforms.blended_transform_factory(
        plot.get_yticklabels()[0].get_transform(), plot.transData)
    plot.text(0.9, -0.025, '{:.0f}'.format(90), color='red', transform=trans,
              va='bottom', ha='center')

    save_plot('regularizer_compare_%s' % check_importance_type)

    if show:
        plt.show()
    plt.close()


def create_importance_plot_compare_bn_parameter(filename: str, importance_names: List[str], check_importance_type: str,
                                                timed_name: bool = False, show: bool = False):
    converted_data: List[List[Any]] = []
    for importance_name in importance_names:
        data: Dict[Any, Any] = load_data(filename, importance_name, timed_name)

        importance_label_name: str = 'without \u03B2, ' if 'nobeta_' in importance_name else 'with \u03B2, '
        importance_label_name += 'initial \u03B3 = 0.0' if 'gammazero_' in importance_name else 'initial \u03B3 = 1.0'

        for percent, percent_data in data.items():
            for importance_type, importance_type_data in percent_data.items():
                if importance_type == check_importance_type:
                    plot_point: List[Any] = [importance_label_name, int(percent),
                                             float(importance_type_data['train_accuracy'])]
                    converted_data.append(plot_point)

    df = pd.DataFrame(converted_data,
                      columns=['Importance Generation Method', 'Pruned Edge Percentage', 'Prediction Accuracy'])

    df = df.pivot(index='Pruned Edge Percentage',
                  columns='Importance Generation Method', values='Prediction Accuracy')
    plot: Axes = df.plot(legend=True)
    plot.set_ylabel('Prediction Accuracy')
    plot.set_facecolor('#F4F4F4')
    plot.grid(color='white', linestyle='-', linewidth=2)
    for line in plot.lines:
        plt.setp(line, linewidth=3, alpha=0.6)

    save_plot('bn_parameter_compare_%s' % check_importance_type)

    if show:
        plt.show()
    plt.close()


def create_importance_plot_compare_class_vs_all(filename: str, importance_name: str, class_index: int,
                                                check_importance_type: str, class_specific_data: bool = True,
                                                timed_name: bool = False,
                                                show: bool = False):
    converted_data: List[List[Any]] = []

    importance_data_name: str = '%s_[%s]' % (
        check_importance_type, class_index) if class_specific_data else check_importance_type
    overall_importance_label_name: str = 'all_classes'
    class_importance_label_name: str = 'class_[%s]' % class_index
    data: Dict[Any, Any] = load_data(filename, importance_name, timed_name)

    for percent, percent_data in data.items():
        for importance_type, importance_type_data in percent_data.items():
            if importance_type == importance_data_name:
                plot_point: List[Any] = [overall_importance_label_name, int(percent),
                                         float(importance_type_data['train_accuracy'])]
                converted_data.append(plot_point)
                plot_point = [class_importance_label_name, int(percent),
                              float(importance_type_data['train_class_accuracy'][str(class_index)])]
                converted_data.append(plot_point)

    df = pd.DataFrame(converted_data,
                      columns=['Importance Generation Method', 'Pruned Edge Percentage', 'Prediction Accuracy'])

    df = df.pivot(index='Pruned Edge Percentage',
                  columns='Importance Generation Method', values='Prediction Accuracy')
    plot: Axes = df.plot(legend=True)
    plot.set_ylabel('Prediction Accuracy')

    save_plot('%s_class_compare_%s_%i' %
              (importance_name, check_importance_type, class_index))

    if show:
        plt.show()
    plt.close()


def create_importance_plot_compare_classes_vs_all(filename: str, importance_name: str, check_importance_type: str,
                                                  class_specific_data: bool = True, timed_name: bool = False,
                                                  show: bool = False):
    converted_data: List[List[Any]] = []

    for i in range(10):
        importance_data_name: str = '%s_[%s]' % (
            check_importance_type, i) if class_specific_data else check_importance_type
        class_importance_label_name: str = "Digit \"%s\"" % i
        data: Dict[Any, Any] = load_data(filename, importance_name, timed_name)

        for percent, percent_data in data.items():
            if int(percent) != 100:
                for importance_type, importance_type_data in percent_data.items():
                    if importance_type == importance_data_name:
                        plot_point = [class_importance_label_name, int(percent),
                                      float(importance_type_data['test_class_accuracy'][str(i)]) - float(
                                          importance_type_data['test_accuracy'])]
                        converted_data.append(plot_point)

    df = pd.DataFrame(converted_data,
                      columns=['Importance Generation Method', 'Pruned Edge Percentage', 'Prediction Accuracy'])

    df = df.pivot(index='Pruned Edge Percentage',
                  columns='Importance Generation Method', values='Prediction Accuracy')
    plot: Axes = df.plot(legend=True)
    plot.set_ylabel('Relative Accuracy')
    plt.legend(facecolor='white', framealpha=1)

    plot.set_facecolor('#F4F4F4')
    plot.grid(color='white', linestyle='-', linewidth=2)
    for line in plot.lines:
        plt.setp(line, linewidth=3, alpha=0.7)

    save_plot('%s_class_compare_%s' % (importance_name, check_importance_type))

    if show:
        plt.show()
    plt.close()
