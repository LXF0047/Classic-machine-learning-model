import pandas_profiling
from logging import getLogger, basicConfig, INFO

basicConfig(filename='/home/lxf/projects/auto_boosting/logs/test.log', level=INFO)
logger = getLogger(__name__)


def eda(df, _path, filename='eda'):
    output_path = _path + '%s.html' % filename
    profile = pandas_profiling.ProfileReport(df, minimal=True)
    profile.to_file(output_file=output_path)


def log_evaluation(period=1, show_stdv=True):
    """Create a callback that logs evaluation result with logger.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that logs evaluation every period iterations into logger.
    """

    def _fmt_metric(value, show_stdv=True):
        """format metric string"""
        if len(value) == 2:
            return '%s:%g' % (value[0], value[1])
        elif len(value) == 3:
            if show_stdv:
                return '%s:%g+%g' % (value[0], value[1], value[2])
            else:
                return '%s:%g' % (value[0], value[1])
        else:
            raise ValueError("wrong metric value")

    def callback(env):
        if env.rank != 0 or len(env.evaluation_result_list) == 0 or period is False:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s\n' % (i, msg))

    return callback