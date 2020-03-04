# %% md
# The function get_sum_metrics takes two arguments: a prediction and a list of
# metrics to apply to the prediction (say, for instance, the accuracy or the
# precision). Note that each metric is a function, not a number. The function
# should compute each of the metrics for the prediction and sum them. It should
# also add to this sum three default metrics, in this case, adding 0, 1 or 2 to
# the prediction.


# %%
def get_sum_metrics(predictions, metrics=None):

    if metrics is None:
        metrics = []

    for i in range(3):
        metrics.append(lambda x, i=i: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics
