import numpy as np


class Statistics(object):
    """
    Base class for all statistical methods to be applied to data collected.
    """
    display_name = "Statistics"

    @staticmethod
    def compute(data):
        """
        Compute the statistics and returns the computed value.

        :param data: The data to be computed.
        """
        raise NotImplementedError


class StatNone(Statistics):
    """
    Empty Statistic.
    """
    display_name = "None"

    @staticmethod
    def compute(data):
        """
        Don't compute a statistic, just return the data as is.

        :return: data
        """
        return data


class StatMedian(Statistics):
    display_name = "Median"

    @staticmethod
    def compute(data):
        return np.median(data)


class StatStdDeviation(Statistics):
    display_name = "Standard Deviation"

    @staticmethod
    def compute(data):
        return np.std(data)


class StatMedianDeviation(Statistics):
    display_name = "Median Deviation"

    @staticmethod
    def compute(data):
        median = np.median(data)
        return np.median(np.abs(data - median))


class StatMax(Statistics):
    display_name = "Max"

    @staticmethod
    def compute(data):
        return np.max(data)


class StatMin(Statistics):
    display_name = "Min"

    @staticmethod
    def compute(data):
        return np.min(data)


class Stat80Percent(Statistics):
    display_name = "80th percentile"

    @staticmethod
    def compute(data):
        return np.percentile(data, 80)


class StatAvgMean(Statistics):
    display_name = "Avg. of points > mean"

    @staticmethod
    def compute(data):
        percentile = np.percentile(data, 50)
        return np.mean(data[data > percentile])


class Stat20Percent(Statistics):
    display_name = "20th percentile"

    @staticmethod
    def compute(data):
        return np.percentile(data, 20)


class StatMean(Statistics):
    display_name = "Mean"

    @staticmethod
    def compute(data):
        return np.mean(data)


all_stats = [StatNone, StatMedian, StatStdDeviation, StatMedianDeviation, StatMax, StatMin, Stat80Percent,
             StatAvgMean, Stat20Percent, StatMean]
