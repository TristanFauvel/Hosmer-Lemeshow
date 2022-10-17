import pandas as pd
import numpy as np
from scipy.stats import chi2


def HosmerLemeshow(data, predictions, Q=10):
    """Hosmer-Lemeshow goodness of fit test


    Parameters
    ----------
    data : Dataframe
        Dataframe containing :
            - the effective of each group n
            - for each group, the number of cases where c = 1, n1
    predictions : Dataframe
        Dataframe containing :
            - for each group, the predicted proportion of c = 1 according to a logistic regression, p1
    Q : int, optional
        The number of groups

    Returns
    -------
    result : the result of the test, including Chi2-HL statistics and p-value 

    """

    data.loc[:, "expected_prop"] = predictions.p1
    data = data.sort_values(by="expected_prop", ascending=False)

    categories, bins = pd.cut(
        data["expected_prop"],
        np.percentile(data["expected_prop"], np.linspace(0, 100, Q + 1)),
        labels=False,
        include_lowest=True,
        retbins=True,
    )

    meanprobs1 = np.zeros(Q)
    expevents1 = np.zeros(Q)
    obsevents1 = np.zeros(Q)
    meanprobs0 = np.zeros(Q)
    expevents0 = np.zeros(Q)
    obsevents0 = np.zeros(Q)

    y0 = data.n - data.n1
    y1 = data.n1
    n = data.n
    expected_prop = predictions.p1
    for i in range(Q):

        meanprobs1[i] = np.mean(expected_prop[categories == i])
        expevents1[i] = np.sum(n[categories == i]) * np.array(meanprobs1[i])
        obsevents1[i] = np.sum(y1[categories == i])
        meanprobs0[i] = np.mean(1 - expected_prop[categories == i])
        expevents0[i] = np.sum(n[categories == i]) * np.array(meanprobs0[i])
        obsevents0[i] = np.sum(y0[categories == i])

    data1 = {"meanprobs1": meanprobs1, "meanprobs0": meanprobs0}
    data2 = {"expevents1": expevents1, "expevents0": expevents0}
    data3 = {"obsevents1": obsevents1, "obsevents0": obsevents0}
    m = pd.DataFrame(data1)
    e = pd.DataFrame(data2)
    o = pd.DataFrame(data3)

    chisq_value = sum(sum((np.array(o) - np.array(e)) ** 2 / np.array(e)))
    pvalue = 1 - chi2.cdf(chisq_value, Q - 2)

    result = pd.DataFrame(
        [[Q - 2, chisq_value.round(2), pvalue.round(2)]],
        columns=["df", "Chi2", "p - value"],
    )
    return result
