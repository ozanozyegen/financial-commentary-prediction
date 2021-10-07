import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

def get_CAM(model_r, x_test):
    get_last_conv = K.function([model_r.layers[0].input], [model_r.layers[-3].output])
    last_conv = get_last_conv([x_test])[0]

    get_softmax = K.function([model_r.layers[0].input], [model_r.layers[-1].output])
    softmax = get_softmax([x_test])[0]
    softmax_weight = model_r.get_weights()[-2]
    CAM = np.dot(last_conv, softmax_weight)
    CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
    c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
    return c

def get_indexes(y_test,classes):
    """ Returns the indexes from the test set where the label is equal to each class"""
    indexes = []
    for i in range(classes):
        indexes.append(np.array(np.where(y_test == i)))
    return indexes

def get_summary(index,CAM,y_test):
    summary_0 = []
    for i in (index):
        for j in (i):
            summary_0.append(CAM[j,:,int(y_test[j])])
    summary_0 = np.array(summary_0)
    summary_0_sum = np.sum(summary_0,axis = 0)
    return summary_0_sum

def CAM_analysis(model, x_test, y_test):
    classes = len(np.unique(y_test))
    indexes = get_indexes(y_test, classes)
    CAM = get_CAM(model, x_test)
    # for index in indexes:
    #     print(index)
    sums = [get_summary(index, CAM, y_test) for index in indexes]
    return sums

def cam_graph(model, x_test, y_test, features:list, config):
    """ Generates CAM values """
    feat_cams = CAM_analysis(model, x_test, y_test)

    cam_dict = {}
    for feature, feat_cam in zip(features, feat_cams):
        cam_dict[f'{feature}_mean'] = np.mean(np.array(feat_cam), axis=0)
        cam_dict[f'{feature}_real'] = feat_cam / cam_dict[f'{feature}_mean']
    
    cam_real_dict={k:v for k,v in cam_dict.items() if '_real' in k}
    cam_real_df = pd.DataFrame(cam_real_dict)
    data_cols = [f'{i}' for i in range(-config['HISTORY_SIZE'],0)]
    cam_real_df.index = data_cols
    results = np.array(cam_real_df.values)

    # programmers = features
    # z = np.round(results,3)
    
    # fig = go.Figure(data=go.Heatmap(
    #         z=z,
    #         x=programmers,
    #         y=data_cols,
    #         colorscale='blues'))
    # fig.update_layout(width=600, height=600, xaxis_showgrid=False, yaxis_showgrid=False, template='none')
    
    # fig.update_layout(
    #     title='Contribution of Months in Prediction for CAM')
    
    # fig.show()
    return results
