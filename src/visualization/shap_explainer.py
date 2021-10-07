import shap
import numpy as np

def get_kernel_shap_feature_importances(model, x, nsamples=100):
    samples = shap.sample(x, nsamples, random_state=0)
    e = shap.KernelExplainer(model, samples, link='logit')
    shap_val = np.array(e.shap_values(samples))
    return shap_val

def get_deepshap_feature_importances(model, x, nsamples=100):
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    samples = shap.sample(x, nsamples, random_state=0)
    e = shap.DeepExplainer(model, samples)
    shap_val = np.array(e.shap_values(samples))
    return shap_val