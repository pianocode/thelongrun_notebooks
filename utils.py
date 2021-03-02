from urllib.request import urlretrieve
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def downloadFromUrl(url: str, outputFile: str = None) -> None:
    if outputFile is None:
        outputFile = url.split(sep="/")[-1]
    else:
        if len(outputFile.split(".")) != 2:
            ext = url.split(".")[-1]
            outputFile = ".".join([outputFile, ext])
            print("Added '." + ext + "' to the output file name.")
    if not os.path.exists(outputFile):
        urlretrieve(url, outputFile)


def computeLoglikelihood(mean, std, data):
    """
    Compute log-likelihood of the data given each combination of mean and std
    """
    log_likelihood = tf.map_fn(
        fn=lambda x: tfd.Independent(
            tfd.Normal(mean, std), reinterpreted_batch_ndims=0).log_prob(x),
        elems=data)
    # Sum the log-likelihood for each point in data i.i.d.
    total_log_likelihood = tf.reduce_sum(log_likelihood, axis=0)
    return total_log_likelihood


@tf.function(experimental_compile=True)
def posteriorMesh(mean_vector, std_vector, mean_d, std_d, data):
    M, S = tf.meshgrid(mean_vector, std_vector)
    prob_M = mean_d.log_prob(M)
    prob_S = std_d.log_prob(S)
    if data.dtype != tf.float32:
        data = tf.cast(data, tf.float32)
    prob_LL = computeLoglikelihood(M, S, data)
    return prob_M + prob_S + prob_LL


def normalizePosterior(prob):
    return tf.exp(prob - tf.reduce_max(prob))
