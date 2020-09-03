import pandas_profiling


def eda(df, _path, filename='eda'):
    output_path = _path + '%s.html' % filename
    profile = pandas_profiling.ProfileReport(df, minimal=True)
    profile.to_file(output_file=output_path)